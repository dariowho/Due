import unittest

from due.nlp.vocabulary import *

class TestVocabulary(unittest.TestCase):

	def test_add_word(self):
		v = Vocabulary()

		old_size = v.size()

		v.add_word('foo')
		v.add_word('bar')

		self.assertEqual(v.size(), old_size+2)

		foo_index = v.index('foo')
		bar_expected_index = foo_index + 1
		self.assertEqual(v.index('bar'), bar_expected_index)
		self.assertEqual(v.word(foo_index), 'foo')
		self.assertEqual(v.word(bar_expected_index), 'bar')

	def test_unknown_words(self):
		v = Vocabulary()
		v.add_word('foo')

		unk_index = v.index(UNK)

		self.assertNotEqual(v.index('foo'), unk_index)
		self.assertEqual(v.index('nonexistentword'), unk_index)

	def test_load_saved(self):
		v = Vocabulary()
		v.add_word('foo')
		v.add_word('foo')
		v.add_word('bar')

		v_saved = v.save()
		v_loaded = Vocabulary.load(v_saved)

		self.assertEqual(v.size(), v_loaded.size())
		self.assertEqual(v.index('foo'), v_loaded.index('foo'))
		self.assertEqual(v.index('bar'), v_loaded.index('bar'))
		self.assertEqual(v.index_to_count[v.index('foo')], v_loaded.index_to_count[v_loaded.index('foo')])

		v.add_word('asd')
		v_loaded.add_word('asd')
		self.assertEqual(v.index('asd'), v_loaded.index('asd'))

	def test_prune_vocabulary(self):
		v = Vocabulary()
		v.add_word('foo')
		v.add_word('foo')
		v.add_word('foo')
		v.add_word('bar')
		v.add_word('bar')

		v_pruned = prune_vocabulary(v, 3)

		self.assertEqual(v_pruned.size(), v.size()-1)
		self.assertEqual(v_pruned.index('bar'), v_pruned.index(UNK))
		self.assertIn('foo', v.word_to_index)
		self.assertIn('foo', v_pruned.word_to_index)
		self.assertIn('bar', v.word_to_index)
		self.assertNotIn('bar', v_pruned.word_to_index)

		bar_index = v.index('bar')
		self.assertIn(bar_index, v.index_to_word)
		self.assertIn(bar_index, v.index_to_count)
		self.assertNotIn(bar_index, v_pruned.index_to_count)
		self.assertNotIn(bar_index, v_pruned.index_to_count)
