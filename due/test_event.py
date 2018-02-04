import unittest

from due.agent import HumanAgent
from due.event import *

from datetime import datetime

class TestEvent(unittest.TestCase):

	def test_mark_acted(self):
		e1 = Event(Event.Type.Utterance, None, None, "hello there")
		self.assertIsNone(e1.acted)
		now = datetime.now()
		e1.mark_acted()
		self.assertIsNotNone(e1.acted)
		self.assertGreaterEqual(e1.acted, now)
		self.assertLessEqual(e1.acted, datetime.now())

		e1 = Event(Event.Type.Utterance, None, None, "hello there")
		e1.mark_acted(datetime(2018, 2, 4, 18, 5, 25, 261308))
		self.assertEqual(e1.acted, datetime(2018, 2, 4, 18, 5, 25, 261308))

	def test_clone(self):
		e1 = Event(Event.Type.Utterance, None, None, "hello there")
		e2 = e1.clone()
		self.assertEqual(e1, e2)

		e1 = Event(Event.Type.Utterance, None, None, "hello there")
		e1.mark_acted()
		e2 = e1.clone()
		self.assertEqual(e1, e2)
		self.assertIsNone(e2.acted)

	def test_equal(self):
		a = HumanAgent('Alice')
		e0 = Event(Event.Type.Utterance, None, None, "hello there")
		e1 = Event(Event.Type.Utterance, None, None, "hello there")
		e2 = Event(Event.Type.Action, None, None, "hello there")
		e3 = Event(Event.Type.Utterance, None, None, "general Kenobi!")
		e4 = Event(Event.Type.Utterance, datetime.now(), None, "hello there")
		e5 = Event(Event.Type.Utterance, None, a, "hello there")

		self.assertEqual(e0, e1)
		self.assertNotEqual(e0, e3)
		self.assertNotEqual(e0, e4)
		self.assertNotEqual(e0, e5)

	def test_save(self):
		a = HumanAgent('Alice')
		now = datetime.now()
		e = Event(Event.Type.Utterance, now, a, "hello there")
		saved = e.save()
		self.assertEqual(saved[0], Event.Type.Utterance.value)
		self.assertEqual(saved[1], now.isoformat())
		self.assertEqual(saved[2], a.id)
		self.assertEqual(saved[3], 'hello there')

