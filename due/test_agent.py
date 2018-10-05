import unittest

from datetime import datetime
import tempfile
import os

from due.persistence import serialize, deserialize
from due.agent import *
from due.action import Action, RecordedAction
from due.event import Event

class TestAgent(unittest.TestCase):

	def test_agent_episode(self):
		ha_alice = HumanAgent(name='Alice')
		ha_bob = HumanAgent(name='Bob')
		e = ha_alice.start_episode(ha_bob)
		self.assertEqual(len(e.events), 0)

		utterance = Event(Event.Type.Utterance, datetime.now(), ha_alice.id, "alice1")
		ha_alice.act_events([utterance], e)
		self.assertEqual(e.events[-1].payload, 'alice1')

		utterance = Event(Event.Type.Utterance, datetime.now(), ha_alice.id, "alice2")
		ha_alice.act_events([utterance], e)
		self.assertEqual(len(e.events), 2)
		self.assertEqual(e.events[-1].payload, 'alice2')

		utterance = Event(Event.Type.Utterance, datetime.now(), ha_bob.id, "bob1")
		ha_bob.act_events([utterance], e)
		self.assertEqual(len(e.events), 3)
		self.assertEqual(e.events[-1].payload, 'bob1')

		action = Event(Event.Type.Action, datetime.now(), ha_alice.id, RecordedAction())
		ha_alice.act_events([action], e)
		self.assertTrue(action.payload.done)
		self.assertEqual(len(e.events), 4)
		self.assertTrue(e.events[-1].payload is action.payload)

		action = RecordedAction()
		events = [Event(Event.Type.Utterance, datetime.now(), ha_alice.id, "alice3"),
		          Event(Event.Type.Action, datetime.now(), ha_alice.id, action)]
		ha_alice.act_events(events, e)
		self.assertTrue(action.done)
		self.assertEqual(len(e.events), 6)
		self.assertEqual(e.events[-2].payload, 'alice3')
		self.assertTrue(e.events[-1].payload is action)


	def test_agent_episode_deprecated(self):
		ha_alice = HumanAgent(name='Alice')
		ha_bob = HumanAgent(name='Bob')
		e = ha_alice.start_episode(ha_bob)
		self.assertEqual(len(e.events), 0)

		ha_alice.say('alice1', e)
		self.assertEqual(len(e.events), 1)
		self.assertEqual(e.events[-1].payload, 'alice1')

		ha_alice.say('alice2', e)
		self.assertEqual(len(e.events), 2)
		self.assertEqual(e.events[-1].payload, 'alice2')

		ha_bob.say('bob1', e)
		self.assertEqual(len(e.events), 3)
		self.assertEqual(e.events[-1].payload, 'bob1')

		action = RecordedAction()
		ha_alice.do(action, e)
		self.assertTrue(action.done)
		self.assertEqual(len(e.events), 4)
		self.assertTrue(e.events[-1].payload is action)

		ha_alice.leave(e)


	def test_human_agent_load_save(self):
		ha_alice = HumanAgent(name='Alice')

		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_human_agent_load_save.pkl')

		serialize(ha_alice.save(), test_path)
		loaded_alice = HumanAgent.load(deserialize(test_path))
		self.assertEqual(ha_alice.id, loaded_alice.id)
		self.assertEqual(ha_alice.name, loaded_alice.name)
