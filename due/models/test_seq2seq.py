import unittest
from datetime import datetime
import tempfile
import os

import torch

from due.brain import Brain
from due.episode import Episode
from due.event import Event
from due.persistence import serialize, deserialize
from due.models.seq2seq import EncoderDecoderBrain

class TestEncoderDecoderBrain(unittest.TestCase):

	def test_save_load(self):
		brain = EncoderDecoderBrain({
			'batch_size': 1,
			'hidden_size': 16,
		}, _get_train_episodes(), random_embedding_init=True)
		saved_brain = brain.save()

		with tempfile.TemporaryDirectory() as temp_dir:
			path = os.path.join(temp_dir, 'serialized_encdec_brain.due')
			serialize(saved_brain, path)
			loaded_brain = Brain.load(deserialize(path))

		self.assertEqual(brain.X, loaded_brain.X)
		self.assertEqual(brain.y, loaded_brain.y)
		self.assertEqual(brain.vocabulary.save(), loaded_brain.vocabulary.save())
		self.assertEqual(brain.parameters, loaded_brain.parameters)
		assert torch.all(torch.eq(brain.embedding_matrix, loaded_brain.embedding_matrix))

		loaded_encoder_state = loaded_brain.encoder.state_dict()
		for k, v in brain.encoder.state_dict().items():
			assert torch.all(torch.eq(v, loaded_encoder_state[k]))

		loaded_decoder_state = loaded_brain.decoder.state_dict()
		for k, v in brain.decoder.state_dict().items():
			assert torch.all(torch.eq(v, loaded_decoder_state[k]))

	def test_reset_with_parameters(self):
		brain = EncoderDecoderBrain({
			'batch_size': 1,
			'hidden_size': 16,
		}, _get_train_episodes(), random_embedding_init=True)

		brain_new = brain.reset_with_parameters({'hidden_size': 8})
		self.assertEqual(brain.X, brain_new.X)
		self.assertEqual(brain.y, brain_new.y)
		self.assertEqual(brain.vocabulary.save(), brain_new.vocabulary.save())
		assert torch.all(torch.eq(brain.embedding_matrix, brain_new.embedding_matrix))
		self.assertEqual(brain_new.parameters, {**brain_new.parameters, **{'hidden_size': 8}})
		self.assertEqual(brain.encoder.gru.hidden_size, 16)
		self.assertEqual(brain_new.encoder.gru.hidden_size, 8)
		self.assertEqual(brain.decoder.gru.hidden_size, 16)
		self.assertEqual(brain_new.decoder.gru.hidden_size, 8)

	def test_epoch_predict(self):
		brain = EncoderDecoderBrain({
			'batch_size': 2,
			'hidden_size': 16,
		}, _get_train_episodes(), random_embedding_init=True)
		brain.epoch()
		brain.predict('just an utterance')

	def test_long_utterances(self):
		brain = EncoderDecoderBrain({
			'batch_size': 2,
			'hidden_size': 16,
			'max_sentence_length': 2
		}, _get_train_episodes(), random_embedding_init=True)
		brain.epoch()
		brain.predict('just an utterance')

def _get_train_episodes():
	result = []

	e = Episode('a', 'b')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), 'a', 'this is the first utterance'),
		Event(Event.Type.Utterance, datetime.now(), 'b', 'this is the first answer'),
		Event(Event.Type.Utterance, datetime.now(), 'a', 'this one is the second utterance'),
		Event(Event.Type.Utterance, datetime.now(), 'b', 'this is the second answer')
	]
	result.append(e)

	e = Episode('1', '2')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), '1', 'one one one'),
		Event(Event.Type.Utterance, datetime.now(), '2', 'two two two'),
		Event(Event.Type.Utterance, datetime.now(), '1', 'three three three'),
		Event(Event.Type.Utterance, datetime.now(), '2', 'four four four')
	]
	result.append(e)

	return result

def _get_test_episode():
	e = Episode('a', 'b')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
	]
	return e
