import unittest

from numpy.testing import assert_array_equal

from due.nlp.batches import batches, batch_to_matrix, batch_to_tensor, pad_sequence
from due.nlp.vocabulary import Vocabulary, EOS, UNK

class TestBatches(unittest.TestCase):

	def test_batches(self):
		X = [0, 1, 2, 3, 4, 5]
		y = ['a', 'b', 'c', 'd', 'e', 'f']
		result = list(batches(X, y, 3))
		self.assertEqual(result, [
			([0, 1, 2], ['a', 'b', 'c']),
			([3, 4, 5], ['d', 'e', 'f']),
		])

		result = list(batches(X, y, 4))
		self.assertEqual(result, [
			([0, 1, 2, 3], ['a', 'b', 'c', 'd']),
			([4, 5], ['e', 'f']),
		])  


class TestPadSequence(unittest.TestCase):

	def test_cut_shorter(self):
		s = [1, 2, 3, 4, 5]
		result = pad_sequence(s, 0, 3)
		self.assertEqual(result, [1, 2, 0])

	def test_same_length(self):
		s = [1, 2, 3, 4, 5]
		result = pad_sequence(s, 0, 5)
		self.assertEqual(result, [1, 2, 3, 4, 0])

	def test_longer(self):
		s = [1, 2, 3, 4, 5]
		result = pad_sequence(s, 0, 8)
		self.assertEqual(result, [1, 2, 3, 4, 5, 0, 0, 0])

class TestBatchToMatrix(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.v = Vocabulary()
		self.v.add_word('aaa')
		self.v.add_word('bbb')
		self.v.add_word('ccc')

		self.aaa = self.v.index('aaa')
		self.bbb = self.v.index('bbb')
		self.ccc = self.v.index('ccc')
		self.EOS = self.v.index(EOS)
		self.UNK = self.v.index(UNK)

	def test_batch_to_matrix(self):
		batch = ['aaa aaa bbb', 'ccc aaa bbb']
		m = batch_to_matrix(batch, self.v)

		assert_array_equal(m, [
			[[self.aaa], [self.ccc]],
			[[self.aaa], [self.aaa]],
			[[self.bbb], [self.bbb]],
			[[self.EOS], [self.EOS]],
		])

	def test_different_lengths(self):
		batch = ['aaa', 'aaa bbb', 'ccc ccc ccc ccc']
		m = batch_to_matrix(batch, self.v)
		assert_array_equal(m, [
			[[self.aaa], [self.aaa], [self.ccc]],
			[[self.EOS], [self.bbb], [self.ccc]],
			[[self.EOS], [self.EOS], [self.ccc]],
			[[self.EOS], [self.EOS], [self.ccc]],
			[[self.EOS], [self.EOS], [self.EOS]],
		])

	def test_unknown_words(self):
		batch = ['aaa ddd', 'aaa bbb']
		m = batch_to_matrix(batch, self.v)
		assert_array_equal(m, [
			[[self.aaa], [self.aaa]],
			[[self.UNK], [self.bbb]],
			[[self.EOS], [self.EOS]],
		])


	def test_max_words(self):
		batch = ['aaa', 'aaa bbb', 'ccc ccc ccc ccc']
		m = batch_to_matrix(batch, self.v, max_words=2)
		assert_array_equal(m, [
			[[self.aaa], [self.aaa], [self.ccc]],
			[[self.EOS], [self.bbb], [self.ccc]],
			[[self.EOS], [self.EOS], [self.EOS]],
		])
