import unittest

from datetime import datetime
import tempfile
import os

from due.persistence import serialize, deserialize
from due.models.dummy import DummyAgent
from due.action import RecordedAction
from due.event import Event

class TestAgent(unittest.TestCase):
	def setUp(self):
		self.alice = DummyAgent("Alice")
		self.bob = DummyAgent("Bob")
		self.sample_episode = self.alice.start_episode(self.bob)
		self.alice.say("Hi!", self.sample_episode)
		self.bob.say("Hello", self.sample_episode)
		self.alice.say("How are you?", self.sample_episode)
		self.bob.say("Good thanks, and you?", self.sample_episode)
		self.alice.say("All good", self.sample_episode)

	def test_agent_episode(self):
		e = self.alice.start_episode(self.bob)
		self.assertEqual(len(e.events), 0)

		utterance = Event(Event.Type.Utterance, datetime.now(), self.alice.id, "alice1")
		self.alice.act_events([utterance], e)
		self.assertEqual(e.events[-1].payload, 'alice1')

		utterance = Event(Event.Type.Utterance, datetime.now(), self.alice.id, "alice2")
		self.alice.act_events([utterance], e)
		self.assertEqual(len(e.events), 2)
		self.assertEqual(e.events[-1].payload, 'alice2')

		utterance = Event(Event.Type.Utterance, datetime.now(), self.bob.id, "bob1")
		self.bob.act_events([utterance], e)
		self.assertEqual(len(e.events), 3)
		self.assertEqual(e.events[-1].payload, 'bob1')

		action = Event(Event.Type.Action, datetime.now(), self.alice.id, RecordedAction())
		self.alice.act_events([action], e)
		self.assertTrue(action.payload.done)
		self.assertEqual(len(e.events), 4)
		self.assertTrue(e.events[-1].payload is action.payload)

		action = RecordedAction()
		events = [Event(Event.Type.Utterance, datetime.now(), self.alice.id, "alice3"),
		          Event(Event.Type.Action, datetime.now(), self.alice.id, action)]
		self.alice.act_events(events, e)
		self.assertTrue(action.done)
		self.assertEqual(len(e.events), 6)
		self.assertEqual(e.events[-2].payload, 'alice3')
		self.assertTrue(e.events[-1].payload is action)


	def test_agent_episode_deprecated(self):
		e = self.alice.start_episode(self.bob)
		self.assertEqual(len(e.events), 0)

		self.alice.say('alice1', e)
		self.assertEqual(len(e.events), 1)
		self.assertEqual(e.events[-1].payload, 'alice1')

		self.alice.say('alice2', e)
		self.assertEqual(len(e.events), 2)
		self.assertEqual(e.events[-1].payload, 'alice2')

		self.bob.say('bob1', e)
		self.assertEqual(len(e.events), 3)
		self.assertEqual(e.events[-1].payload, 'bob1')

		action = RecordedAction()
		self.alice.do(action, e)
		self.assertTrue(action.done)
		self.assertEqual(len(e.events), 4)
		self.assertTrue(e.events[-1].payload is action)

		self.alice.leave(e)

	def test_human_agent_load_save(self):
		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_human_agent_load_save.pkl')

		serialize(self.alice.save(), test_path)
		loaded_alice = DummyAgent.load(deserialize(test_path))
		self.assertEqual(self.alice.id, loaded_alice.id)

