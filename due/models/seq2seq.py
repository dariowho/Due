"""
This module defines a very basic neural model for dialogue management. This
model is loosely inspired to the one defined in [1], which is in turn an
application of the **seq2seq** framework ([2]) to dialogue management.

The model features a single-layer GRU encoder and a similarly defined decoder.

Because of its affinity with machine-translation approaches, most of the code
below is taken from this tutorial by Sean Robertson:

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

The model is implemented using the **PyTorch** framework.

[1] Vinyals, O. & Le, Q. V. (2015). A Neural Conversational Model.. CoRR, abs/1506.05869.
[2] Sutskever, I., Vinyals, O. & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in neural information processing systems (p./pp. 3104--3112), . 
"""

from __future__ import unicode_literals, print_function, division

import logging
import random
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import due
from due.brain import Brain
from due.event import Event
from due.episode import extract_utterance_pairs
from due.nlp.vocabulary import Vocabulary, get_embedding_matrix, prune_vocabulary, SOS, EOS
from due.nlp.preprocessing import normalize_sentence
from due.nlp.batches import batches, pad_sequence, batch_to_tensor
from due.util.python import is_notebook
from due import resource_manager
rm = resource_manager

if is_notebook():
	from tqdm import tqdm_notebook as tqdm
else:
	import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# Model definition
#

class EncoderRNNBatch(nn.Module):
	def __init__(self, hidden_size, embedding_matrix, num_rnn_layers=1):
		super(EncoderRNNBatch, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
		embedding_dim = self.embedding.embedding_dim

		self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_rnn_layers)

		self._num_rnn_layers = num_rnn_layers

	def forward(self, input_data, batch_size, hidden):
		embedded = self.embedding(input_data).view(1, batch_size, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def init_hidden(self, batch_size):
		return torch.zeros(self._num_rnn_layers, batch_size, self.hidden_size, device=DEVICE)

class DecoderRNNBatch(nn.Module):
	def __init__(self, hidden_size, embedding_matrix, num_rnn_layers=1):
		super(DecoderRNNBatch, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
		embedding_dim = self.embedding.embedding_dim
		vocabulary_size = self.embedding.num_embeddings

		self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_rnn_layers)
		self.out = nn.Linear(hidden_size, vocabulary_size)
		self.softmax = nn.LogSoftmax(dim=1)

		self._num_rnn_layers = num_rnn_layers

	def forward(self, input_data, batch_size, hidden):
		output = self.embedding(input_data).view(1, batch_size, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.out(output[0])
		output = self.softmax(output)
		return output, hidden

	def init_hidden(self, batch_size):
		return torch.zeros(self._num_rnn_layers, batch_size, self.hidden_size, device=DEVICE)

#
# Brain
#

DEFAULT_PARAMETERS = {
	'batch_size': 64,
	'hidden_size': 512,
	'num_rnn_layers': 1,
	'learning_rate': 0.01,
	'max_sentence_length': 20,
	'teacher_forcing_ratio': 1.0
}

class EncoderDecoderBrain(Brain):

	def __init__(self, model_parameters, initial_episodes, vocabulary_min_count=25, _data=None, _dataset_data=None):
		"""
		The `EncoderDecoderBrain` implements the :class:`due.brain.Brain` class
		with the EncoderDecoder framework described in this module.

		Currently, the following **model parameters** are accepted:

		* `batch_size`: batch size for training (default 64)
		* `hidden_dim`: number of hidden cells (default 512)
		* `learning_rate`: gradient descent's learning rate (default 0.01)
		* `max_sentence_length`: sentences longer than this will be trimmed
		  (default 20)
		* `teacher_forcing_ratio`: currently, this must be 1.0

		A fresh `EncoderDecoderBrain` must be provided with a set of **initial
		episodes** to learn. It is especially important to choose carefully the
		initial set of episodes, because they will be used to extract the Brain's
		Vocabulary (that is, the set of words the Brain will be able to
		understand and produce). Vocabularies are **immutable** for the whole
		life of the Brain instance: currently, the only way to add new words to
		an `EncoderDecoderBrain` is to train a new Brain from scratch.

		:param model_parameters: a dictionary of parameters.
		:type model_parameters: `dict`
		:param initial_episodes: a list of Episodes
		:type initial_episodes: `list` of :class:`due.episode.Episode`
		:param vocabulary_min_count: prune words with less than this number occurrences when building the Brain's vocabulary
		:type vocabulary_min_count: `int`
		:param _data: used internally by :meth:`EncoderDecoderBrain.load`
		:type _data: `dict`
		:param _data: used internally by :meth:`EncoderDecoderBrain.reset_with_parameters`
		:type _data: `dict`
		"""

		self._logger = logging.getLogger(__name__)
		if _data:
			self._init_from_data(_data)
		elif _dataset_data:
			self._init_from_dataset(_dataset_data)
		else:
			self._init_from_scratch(model_parameters, initial_episodes, vocabulary_min_count)

		self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.parameters['learning_rate'])
		self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.parameters['learning_rate'])
		self.criterion = nn.NLLLoss()

	def _init_from_data(self, data):
		self._init_from_dataset(data)

		self.encoder.load_state_dict(data['model']['encoder'])
		self.decoder.load_state_dict(data['model']['decoder'])
		self.epochs = data['model']['epochs']
		self.train_loss_history = data['model']['train_loss_history']

	def _init_from_dataset(self, data):
		self.parameters = data['parameters']
		self.X = data['dataset']['X']
		self.y = data['dataset']['y']
		self.vocabulary = Vocabulary.load(data['dataset']['vocabulary'])
		self.embedding_matrix = data['dataset']['embedding_matrix']

		self._init_model()

	def _init_from_scratch(self, parameters, episodes, vocabulary_min_count):
		self.parameters = {**DEFAULT_PARAMETERS, **parameters}

		self.X = []
		self.y = []
		self._logger.info("Extracting dataset from episodes")
		for e in tqdm(episodes):
			try:
				episode_X, episode_y = extract_utterance_pairs(e, preprocess_f=normalize_sentence)
			except AttributeError:
				self._logger.warning("Skipping episode with events: %s" % e.events)
			self.X.extend(episode_X)
			self.y.extend(episode_y)

		vocabulary_full = Vocabulary()
		for sentence in set(self.X + self.y):
			for word in sentence.split():
				vocabulary_full.add_word(word)
		self.vocabulary = prune_vocabulary(vocabulary_full, min_occurrences=vocabulary_min_count)

		self._logger.info("Building the embedding matrix")
		with rm.open_resource_file('embeddings.glove6B', 'glove.6B.300d.txt') as f:
			self.embedding_matrix = torch.FloatTensor(get_embedding_matrix(self.vocabulary, f, 300), device=DEVICE)

		self._logger.info("Initializing model")
		self._init_model()

	def _init_model(self):
		self.encoder = EncoderRNNBatch(self.parameters['hidden_size'], self.embedding_matrix, self.parameters['num_rnn_layers']).to(DEVICE)
		self.decoder = DecoderRNNBatch(self.parameters['hidden_size'], self.embedding_matrix, self.parameters['num_rnn_layers']).to(DEVICE)
		self.epochs = 0
		self.train_loss_history = []

	def learn_episodes(self, episodes):
		raise NotImplementedError()

	def new_episode_callback(self, episode):
		self._logger.info("New episode started")

	def utterance_callback(self, episode):
		last_utterance = episode.last_event(Event.Type.Utterance)
		return [self.predict(last_utterance.payload)]

	def leave_callback(self, episode, agent):
		self._logger.info(f"Agent {agent} left the episode")

	def predict(self, sentence):
		"""
		Predict an answer for the given `sentence`

		:param sentence: a sentence
		:type sentence: `str`
		"""
		result = []

		sentence = normalize_sentence(sentence)
		input_tensor = batch_to_tensor([sentence], self.vocabulary, device=DEVICE)
		input_length = input_tensor.size(0)
		batch_size = input_tensor.size(1)

		encoder_hidden = self.encoder.init_hidden(batch_size)
		for ei in range(input_length):
			_, encoder_hidden = self.encoder(input_tensor[ei], batch_size, encoder_hidden)

		decoder_input = torch.tensor([[self.vocabulary.index(SOS)] * batch_size], device=DEVICE)
		decoder_hidden = encoder_hidden

		for di in range(self.parameters['max_sentence_length']):
			decoder_output, decoder_hidden = self.decoder(decoder_input, batch_size, decoder_hidden)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()
			
			predicted_index = decoder_input.item()
			
			if predicted_index == self.vocabulary.index(EOS):
				break
			result.append(self.vocabulary.word(predicted_index))

		return " ".join(result)

	def epoch(self):
		"""Run all the episodes encountered do far through training to improve
		the model's predictive capability.

		TODO: allow training on a subset of Episodes
		TODO: allow for an optional validation set
		"""
		tick = datetime.now()
		i = 1
		loss_sum = 0.0
		n_batches = int(np.ceil(len(self.X)/self.parameters['batch_size']))
		progress_iterator = tqdm(batches(self.X, self.y, self.parameters['batch_size']), total=n_batches)
		for input_batch, target_batch in progress_iterator:
			input_tensor = batch_to_tensor(input_batch, self.vocabulary, self.parameters['max_sentence_length'], device=DEVICE)
			target_tensor = batch_to_tensor(target_batch, self.vocabulary, self.parameters['max_sentence_length'], device=DEVICE)

			loss = self._train_batch(input_tensor, target_tensor)
			loss_sum += loss
			average_loss = loss_sum/i
			progress_iterator.set_description("%.4f" % average_loss)
			i += 1
		tock = datetime.now()

		self.train_loss_history.append(average_loss)
		self.epochs += 1

	def _train_batch(self, input_tensor, target_tensor):
		batch_size = input_tensor.size(1)
		input_length = input_tensor.size(0)
		target_length = target_tensor.size(0)

		encoder_hidden = self.encoder.init_hidden(batch_size)

		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()

		for ei in range(input_length):
			encoder_output, encoder_hidden = self.encoder(input_tensor[ei], batch_size, encoder_hidden)

		decoder_input = torch.tensor([[self.vocabulary.index(SOS)]*batch_size], device=DEVICE)
		decoder_hidden = encoder_hidden

		use_teacher_forcing = True if random.random() < self.parameters['teacher_forcing_ratio'] else False

		loss = 0
		if use_teacher_forcing:
			for di in range(target_length):
				decoder_output, decoder_hidden = self.decoder(decoder_input, batch_size, decoder_hidden)
				loss += self.criterion(decoder_output, target_tensor[di].view(batch_size))
				decoder_input = target_tensor[di]
		else:
			raise NotImplementedError()
			# eos_tensor = torch.tensor([vocabulary.index(EOS)], device=DEVICE)
			# for di in range(target_length):
			#     decoder_output, decoder_hidden = self.decoder(decoder_input, batch_size, decoder_hidden)
			#     topv, topi = decoder_output.topk(1)
			#     decoder_input = topi.squeeze().detach()
			#     predicted_words = target_tensor[di].view(batch_size)
			#     loss += self.criterion(decoder_output, predicted_words)
			#     if (predicted_words == eos_tensor*batch_size).all():
			#         break

		loss.backward()

		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

		return loss.item() / target_length

	@staticmethod
	def load(data):
		return EncoderDecoderBrain(None, None, None, _data=data)

	def save(self):
		return {
			'_version': due.__version__,
			'parameters': self.parameters,
			'dataset': self._save_dataset(),
			'model': {
				'encoder': self.encoder.state_dict(),
				'decoder': self.decoder.state_dict(),
				'epochs': self.epochs,
				'train_loss_history': self.train_loss_history
			}
		}

	def reset_with_parameters(self, new_parameters):
		"""
		Return a **new instance** of `EncoderDecoderBrain` with the same data
		as the current one, but with a fresh, randomly initialized, model.
		Different parameters can be specified for the new model.
		"""
		dataset_data = {
			'parameters': {**self.parameters, **new_parameters},
			'dataset': self._save_dataset()
		}
		return EncoderDecoderBrain(None, None, None, _dataset_data=dataset_data)

	def _save_dataset(self):
		return {
			'X': self.X,
			'y': self.y,
			'vocabulary': self.vocabulary.save(),
			'embedding_matrix': self.embedding_matrix
		}