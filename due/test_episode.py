import unittest
import tempfile
import os

from due.persistence import serialize, deserialize
from due.agent import HumanAgent
from due.event import Event
from due.action import RecordedAction
from due.episode import *

from datetime import datetime

class RecordCallbackAgent(HumanAgent):

	def __init__(self, id=None, name=None):
		super().__init__(id, name)
		self.recorded_utterances = 0
		self.recorded_actions = 0
		self.recorded_leave = 0

	def utterance_callback(self, episode):
		self.recorded_utterances += 1
	def action_callback(self, episode):
		self.recorded_actions += 1
	def leave_callback(self, episode):
		self.recorded_leave += 1

class TestEpisode(unittest.TestCase):

	def test_add_event(self):
		alice = HumanAgent('Alice')
		bob = RecordCallbackAgent('Bob')
		episode = alice.start_episode(bob)
		self.assertEqual(bob.recorded_utterances, 0)
		self.assertEqual(bob.recorded_actions, 0)
		self.assertEqual(bob.recorded_leave, 0)
		self.assertEqual(len(episode.events), 0)

		utterance1 = Event(Event.Type.Utterance, datetime.now(), alice.id, 'First utterance')
		episode.add_event(alice, utterance1)
		self.assertEqual(len(episode.events), 1)
		self.assertEqual(episode.events[0], utterance1)
		self.assertIsNotNone(utterance1.acted)
		self.assertEqual(bob.recorded_utterances, 1)
		self.assertEqual(bob.recorded_actions, 0)
		self.assertEqual(bob.recorded_leave, 0)

		utterance2 = Event(Event.Type.Utterance, datetime.now(), bob.id, 'Bob\'s answer')
		episode.add_event(bob, utterance2)
		self.assertEqual(len(episode.events), 2)
		self.assertEqual(episode.events[0], utterance1)
		self.assertEqual(episode.events[1], utterance2)
		self.assertTrue([e.acted is not None for e in episode.events])
		self.assertEqual(bob.recorded_utterances, 1)
		self.assertEqual(bob.recorded_actions, 0)
		self.assertEqual(bob.recorded_leave, 0)

		action1 = Event(Event.Type.Action, datetime.now(), alice.id, RecordedAction())
		episode.add_event(alice, action1)
		self.assertEqual(len(episode.events), 3)
		self.assertEqual(episode.events[0], utterance1)
		self.assertEqual(episode.events[1], utterance2)
		self.assertEqual(episode.events[2], action1)
		self.assertTrue([e.acted is not None for e in episode.events])
		self.assertEqual(bob.recorded_utterances, 1)
		self.assertEqual(bob.recorded_actions, 1)
		self.assertEqual(bob.recorded_leave, 0)

		leave_alice = Event(Event.Type.Leave, datetime.now(), alice.id, None)
		episode.add_event(alice, leave_alice)
		self.assertEqual(len(episode.events), 4)
		self.assertEqual(episode.events[0], utterance1)
		self.assertEqual(episode.events[1], utterance2)
		self.assertEqual(episode.events[2], action1)
		self.assertEqual(episode.events[3], leave_alice)
		self.assertTrue([e.acted is not None for e in episode.events])
		self.assertEqual(bob.recorded_utterances, 1)
		self.assertEqual(bob.recorded_actions, 1)
		self.assertEqual(bob.recorded_leave, 1)

	def test_last_event(self):
		alice = HumanAgent('Alice')
		bob = HumanAgent('Bob')
		episode = alice.start_episode(bob)

		utterance1 = Event(Event.Type.Utterance, datetime.now(), alice.id, 'First utterance')
		action1 = Event(Event.Type.Action, datetime.now(), alice.id, RecordedAction())
		utterance2 = Event(Event.Type.Utterance, datetime.now(), alice.id, 'Second utterance')

		episode.add_event(alice, utterance1)
		episode.add_event(alice, action1)
		episode.add_event(alice, utterance2)

		self.assertEqual(episode.last_event(), utterance2)
		self.assertEqual(episode.last_event(Event.Type.Utterance), utterance2)
		self.assertEqual(episode.last_event(Event.Type.Action), action1)

	def test_empty_episode_save_load(self):
		alice = HumanAgent('Alice')
		bob = HumanAgent('Bob')

		episode = alice.start_episode(bob)
		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_empty_episode_save_load.pkl')
		serialize(episode.save(), test_path)
		loaded_e = Episode.load(deserialize(test_path))

		self.assertEqual(loaded_e.id, episode.id)
		self.assertEqual(loaded_e.starter_id, 'Alice') 
		self.assertEqual(loaded_e.invited_id, 'Bob')
		self.assertEqual(len(loaded_e.events), 0)

	def test_episode_save_load(self):
		alice = HumanAgent('Alice')
		bob = HumanAgent('Bob')
		episode = alice.start_episode(bob)

		utterance1 = Event(Event.Type.Utterance, datetime.now(), alice.id, 'First utterance')
		episode.add_event(alice, utterance1)
		action1 = Event(Event.Type.Action, datetime.now(), alice.id, RecordedAction())
		episode.add_event(alice, action1)
		leave1 = Event(Event.Type.Leave, datetime.now(), alice.id, None)
		episode.add_event(alice, leave1)

		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_episode_save_load.pkl')
		serialize(episode.save(), test_path)
		loaded_e = Episode.load(deserialize(test_path))

		self.assertEqual(loaded_e.starter_id, 'Alice') 
		self.assertEqual(loaded_e.invited_id, 'Bob')
		self.assertEqual(len(loaded_e.events), 3)
		self.assertEqual(loaded_e.events[0], utterance1) 
		self.assertEqual(loaded_e.events[1], action1) 
		self.assertEqual(loaded_e.events[2], leave1) 
