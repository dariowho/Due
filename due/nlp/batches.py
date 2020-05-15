"""
This module contains utility functions to split datasets in batches. This is
typically useful in the context of batched Gradient Descent for the optimization
of Deep Learning models.
"""
import logging
import numpy as np
from due.nlp.vocabulary import EOS

logger = logging.getLogger(__name__)

try:
	import torch
except ImportError:
	logger.warning("Importing 'batching' package with no torch installed. This may cause unexpected behavior")

def batches(X, y, batch_size):
	"""
	Generate two sequences of batches from the input lists `X` and `y`, so that
	each batch contains `batch_size` elements.

	>>> list(batches([0, 1, 2, 3, 4, 5, 6], ['a', 'b', 'c', 'd', 'e', 'f', 'g'], 3))
	[([0, 1, 2], ['a', 'b', 'c']), ([3, 4, 5], ['d', 'e', 'f']), ([6], ['g'])]

	:param X: a sequence of elements
	:type X: `list`
	:param y: a sequence of elements
	:type y: `list`
	:param batch_size: number of elements in each batch
	:type batch_size: `int`
	:return: a generator of the list of batches
	:rtype: `list` of (`list`, `list`)
	"""
	for i in range(int(np.ceil(len(X)/batch_size))):
		start_index = i*batch_size
		end_index = start_index + batch_size
		yield X[start_index:end_index], y[start_index:end_index]

def pad_sequence(sequence, pad_value, final_length):
	"""
	Trim the sequence if longer than final_length, pad it with pad_value if shorter.

	In any case at lest one pad element will be left at the end of the sequence (this is
	because we usually pad with the <EOS> token)

	>>> pad_sequence([1, 2, 3], 0, 5)
	[1, 2, 3, 0, 0]
	>>> pad_sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, 5)
	[1, 2, 3, 4, 0]

	:param sequence: any sequence of elements
	:type sequence: `list`-like
	:param pad_value: a value to pad the sequence with
	:type pad_value: *any*
	:param final_length: length of the final sequence
	:type final_length: `int`
	:return: the padded (or shortened) sequence, with at least one trailing `pad_value`
	:rtype: `list`
	"""
	if len(sequence) >= final_length:
		result = sequence[:final_length]
		result[-1] = pad_value
		return result

	return sequence + [pad_value] * (final_length - len(sequence))

def batch_to_matrix(batch, vocabulary, max_words=None):
	"""
	Receive a list of sentences (strings), return a (*n_words* x *batch_size* x 1)
	matrix `m`, so that `m[i]` contains an array `a` of *batch_size* rows and 1
	column, so that `a[j]` contains the index of the `i`-th word in the `j`-th
	sentence in the batch.

	The **maximum number of words** in the sentences can be limited to
	`max_word`. If `max_words` is not set, the limit will be set by the longest
	sentence in the batch.

	Sentences that are shorter than the maximum length in the resulting matrix
	will be **padded** with EOS. At least one EOS token is appended to every
	sentence in the resulting matrix.

	:param batch: a list of sentences
	:type batch: `list` of `str`
	:param vocabulary: a Vocabulary to look up word indexes
	:type vocabulary: :class:`due.nlp.vocabulary.Vocabulary`
	:param max_words: sentences shorter than `max_words` will be trimmed
	:type max_words: `int`
	:return: a matrix representing the batch
	:rtype: :class:`np.array`
	"""
	sentence_indexes = [[vocabulary.index(w) for w in sentence.split()] for sentence in batch]
	max_length = max([len(x) for x in sentence_indexes])
	if max_words:
		max_length = min(max_length, max_words)
	sentence_indexes = [pad_sequence(s, vocabulary.index(EOS), max_length+1) for s in sentence_indexes]

	result = np.transpose(sentence_indexes)
	result = np.expand_dims(result, axis=2)
	return result

def batch_to_tensor(batch, vocabulary, max_words=None, device=None):
	"""
	Same as :func:`batch_to_matrix`, but returns a Torch tensor, optionally
	mapped to `device`.

	:param batch: a list of sentence
	:type batch: `list` of `str`
	:param vocabulary: a Vocabulary to look up word indexes
	:type vocabulary: :class:`due.nlp.vocabulary.Vocabulary`
	:param max_words: sentences shorter than `max_words` will be trimmed
	:type max_words: `int`
	:param device: a Torch device to map the tensor to (eg. `torch.device("cuda")`)
	:type device: :class:`torch.device`
	:return: a Torch tensor that is equivalent to the output of :func:`batch_to_matrix`
	:rtype: :class:`torch.tensor`
	"""
	result_matrix = batch_to_matrix(batch, vocabulary, max_words)
	return torch.from_numpy(result_matrix, dtype=torch.long).to(device)
