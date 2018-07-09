"""
This module implements a simple :class:`Vocabulary` class to store words and
index them by number.
"""

from collections import defaultdict

import numpy as np
from six import iteritems
from tqdm import tqdm

class Vocabulary():
	def __init__(self):
		self.word_to_index = {}
		self.index_to_word = {}
		self.index_to_count = defaultdict(int)
		self.current_index = 0

		self.add_word('<UNK>') # Unknown token
		self.add_word('<SOS>') # Start of String
		self.add_word('<EOS>') # End of String

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
		Retrieve a word's index in the Vocabulary. Return the index of the <UNK>
		token if not present.

		:param word: the word to look up
		:type word: `str`
		:return: the word's index if existing, *<UNK>*'s index otherwise
		:rtype: `int`
		"""
		if word in self.word_to_index:
			return self.word_to_index[word]
		return self.word_to_index['<UNK>']

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


def prune_vocabulary(vocabulary, min_occurrences):
	"""
	Return a copy of the given vocabulary where words with less than
	`min_occurrences` occurrences are removed.

	:param vocabulary: a Vocabulary
	:type vocabulary: :class:`due.nlp.vocabulary.Vocabulary`
	:param min_occurrences: minimum number of occurrences for a word to be kept
	:type min_occurrences: `int`
	:return: a pruned copy of the given vocabulary
	:rtype: :class:`due.nlp.vocabulary.Vocabulary`
	"""
	result = Vocabulary()
	for index, count in iteritems(vocabulary.index_to_count):
		if count >= min_occurrences:
			result.add_word(vocabulary.word(index))
	return result

def get_embedding_matrix(vocabulary, embeddings_stream, embedding_dim, stub=False):
	"""
	Return a N x D matrix, where N is the number of words in the vocabulary,
	and D is the given embeddings' dimensionality. The *i*-th word in the matrix
	contains the embedding of the word with index *i* in the Vocabulary.

	Sample usage:

	.. code-block:: python

		with rm.open_resource_file('embeddings.glove6B', 'glove.6B.300d.txt') as f:
		    embedding_matrix = get_embedding_matrix(vocabulary, f, 300)

	:param vocabulary: a Vocabulary
	:type vocabulary: :class:`due.nlp.vocabulary.Vocabulary`
	:param embeddings_stream: stream to a resource containing word embeddings in the word2vec format
	:type embeddings_stream: *file*
	:param embedding_dim: dimensionality of the embeddings
	:type embedding_dim: `int`
	:param stub: if True, return a random N x D matrix without reading the embedding source
	:type stub: bool
	"""
	if stub:
		return np.random.rand(vocabulary.size(), embedding_dim)

	unk_index = vocabulary.index('<UNK>')
	result = np.zeros((vocabulary.size(), 300))
	for line in tqdm(embeddings_stream):
		line_split = line.split()
		word = line_split[0]
		index = vocabulary.index(word)
		if index != unk_index:
			vector = [float(x) for x in line_split[1:]]
			result[index,:] = vector
	sos_index = vocabulary.index('<SOS>')
	result[sos_index,:] = np.ones(300)
	return result
