import unittest
import tempfile
import os

from due.persistence import serialize, deserialize
from due.models.dummy import DummyAgent
from due.event import *
from due.action import Action, RecordedAction

from datetime import datetime

T_0 = datetime(2018, 1, 1, 12, 0, 0, 0)


class TestEvent(unittest.TestCase):

	def test_mark_acted(self):
		e1 = Event(Event.Type.Utterance, T_0, None, "hello there")
		self.assertIsNone(e1.acted)
		now = datetime.now()
		e1.mark_acted()
		self.assertIsNotNone(e1.acted)
		self.assertGreaterEqual(e1.acted, now)
		self.assertLessEqual(e1.acted, datetime.now())

		e1 = Event(Event.Type.Utterance, T_0, None, "hello there")
		e1.mark_acted(datetime(2018, 2, 4, 18, 5, 25, 261308))
		self.assertEqual(e1.acted, datetime(2018, 2, 4, 18, 5, 25, 261308))

	def test_clone(self):
		e1 = Event(Event.Type.Utterance, T_0, None, "hello there")
		e2 = e1.clone()
		self.assertEqual(e1, e2)

		e1 = Event(Event.Type.Utterance, T_0, None, "hello there")
		e1.mark_acted()
		e2 = e1.clone()
		self.assertEqual(e1, e2)
		self.assertIsNone(e2.acted)

	def test_equal(self):
		a = DummyAgent('Alice')
		e0 = Event(Event.Type.Utterance, T_0, None, "hello there")
		e1 = Event(Event.Type.Utterance, T_0, None, "hello there")
		e2 = Event(Event.Type.Action, T_0, None, "hello there")
		e3 = Event(Event.Type.Utterance, T_0, None, "general Kenobi!")
		e4 = Event(Event.Type.Utterance, datetime.now(), None, "hello there")
		e5 = Event(Event.Type.Utterance, T_0, a.id, "hello there")

		self.assertEqual(e0, e1)
		self.assertNotEqual(e0, e3)
		self.assertNotEqual(e0, e4)
		self.assertNotEqual(e0, e5)

	def test_event_save(self):
		a = DummyAgent('Alice')
		now = datetime.now()
		e = Event(Event.Type.Utterance, now, a.id, "hello there")

		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_event_save.pkl')
		serialize(e.save(), test_path)
		loaded_e = Event.load(deserialize(test_path))
		self.assertEqual(loaded_e[0], Event.Type.Utterance)
		self.assertEqual(loaded_e[1], now)
		self.assertEqual(loaded_e[2], a.id)
		self.assertEqual(loaded_e[3], 'hello there')

	def test_event_save_action(self):
		"""Save and load an Action event that contains an object payload"""
		a = RecordedAction()
		event = Event(Event.Type.Action, datetime.now(), 'fake-agent-id', a)
		saved_event = event.save()

		loaded_event = Event.load(saved_event)
		assert event == loaded_event
		assert isinstance(loaded_event.payload, Action)
