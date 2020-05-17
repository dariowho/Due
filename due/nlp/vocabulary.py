"""
This module implements a simple :class:`Vocabulary` class to store words and
index them by number.

API
===
"""

from collections import defaultdict

import numpy as np
from six import iteritems

from due.util.python import is_notebook
if is_notebook():
	from tqdm import tqdm_notebook as tqdm
else:
	from tqdm import tqdm

from due import __version__

UNK = '<UNK>'
SOS = '<SOS>'
EOS = '<EOS>'

class Vocabulary():

	"""
	`Vocabulary` is a simple, serializable word index, that comes with utilities
	for its use in Machine Learning experiments.
	"""

	def __init__(self):
		self.word_to_index = {}
		self.index_to_word = {}
		self.index_to_count = defaultdict(int)
		self.current_index = 0

		self.add_word(UNK) # Unknown token
		self.add_word(SOS) # Start of String
		self.add_word(EOS) # End of String

	def add_word(self, word):
		"""
		Add a new word to the dictionary.

		:param word: the word to add
		:type word: `str`
		"""
		if word in self.word_to_index:
			index = self.word_to_index[word]
		else:
			index = self.current_index
			self.current_index += 1
			self.word_to_index[word] = index
			self.index_to_word[index] = word

		self.index_to_count[index] += 1

	def index(self, word):
		"""
		Retrieve a word's index in the Vocabulary. Return the index of the `<UNK>`
		token if not present.

		:param word: the word to look up
		:type word: `str`
		:return: the word's index if existing, `<UNK>`'s index otherwise
		:rtype: `int`
		"""
		if word in self.word_to_index:
			return self.word_to_index[word]
		return self.word_to_index[UNK]

	def word(self, index):
		"""
		Return the word corresponding to the given index

		:param index: the index to look up
		:type index: `int`
		:return: the words corresponding to the given index
		:rtype: `str`
		"""
		return self.index_to_word[index]

	def size(self):
		"""
		Return the number of words in the Vocabulary

		:return: number of words in the Vocabulary
		:rtype: `int`
		"""
		return len(self.word_to_index)

	def save(self):
		"""
		Return a serializable `dict` representing the Vocabulary.

		:return: a serializable representation of self
		:rtype: `dict`
		"""
		return {
			'_version': __version__,
			'word_to_index': self.word_to_index,
	    	'index_to_word': self.index_to_word,
			'index_to_count': self.index_to_count,
			'current_index': self.current_index,
		}

	@staticmethod
	def load(data):
		result = Vocabulary()
		result.word_to_index = data['word_to_index']
		result.index_to_word = data['index_to_word']
		result.index_to_count = data['index_to_count']
		result.current_index = data['current_index']
		return result

def prune_vocabulary(vocabulary, min_occurrences):
	"""
	Return a copy of the given vocabulary where words with less than
	`min_occurrences` occurrences are removed.

	:param vocabulary: a Vocabulary
	:type vocabulary: :class:`Vocabulary`
	:param min_occurrences: minimum number of occurrences for a word to be kept
	:type min_occurrences: `int`
	:return: a pruned copy of the given vocabulary
	:rtype: :class:`Vocabulary`
	"""
	result = Vocabulary()
	for index, count in iteritems(vocabulary.index_to_count):
		if count >= min_occurrences:
			result.add_word(vocabulary.word(index))
	return result

def get_embedding_matrix(vocabulary, embeddings_stream, embedding_dim, random=False):
	"""
	Return a N x D matrix, where N is the number of words in the vocabulary,
	and D is the given embeddings' dimensionality. The *i*-th word in the matrix
	contains the embedding of the word with index *i* in the Vocabulary.

	Sample usage:

	.. code-block:: python

		with rm.open_resource_file('embeddings.glove6B', 'glove.6B.300d.txt') as f:
		    embedding_matrix = get_embedding_matrix(vocabulary, f, 300)

	The *Start Of String* (:data:`SOS`) token is represented as a vector of **ones**.

	:param vocabulary: a Vocabulary
	:type vocabulary: :class:`Vocabulary`
	:param embeddings_stream: stream to a resource containing word embeddings in the word2vec format
	:type embeddings_stream: *file*
	:param embedding_dim: dimensionality of the embeddings
	:type embedding_dim: `int`
	:param random: if True, return a random N x D matrix without reading the embedding source
	:type random: bool
	:return: An embedding matrix for the given vocabulary
	:rtype: :class:`numpy.array`
	"""
	if random:
		return np.random.rand(vocabulary.size(), embedding_dim)

	unk_index = vocabulary.index(UNK)
	result = np.zeros((vocabulary.size(), embedding_dim))
	for line in tqdm(embeddings_stream):
		line_split = line.split()
		word = line_split[0]
		index = vocabulary.index(word)
		if index != unk_index:
			vector = [float(x) for x in line_split[1:]]
			result[index, :] = vector
	sos_index = vocabulary.index(SOS)
	result[sos_index, :] = np.ones(embedding_dim)
	return result
