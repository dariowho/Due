import unittest
import tempfile
import os

from due.persistence import serialize, deserialize
from due.agent import DummyAgent
from due.event import Event
from due.action import RecordedAction
from due.episode import *

from datetime import datetime

class RecordCallbackAgent(DummyAgent):

	def __init__(self):
		super().__init__()
		self.recorded_utterances = 0
		self.recorded_actions = 0
		self.recorded_leave = 0

	def utterance_callback(self, episode, event):
		self.recorded_utterances += 1
	def action_callback(self, episode, event):
		self.recorded_actions += 1
	def leave_callback(self, episode, event):
		self.recorded_leave += 1

class TestEpisode(unittest.TestCase):

	def test_add_event(self):
		alice = DummyAgent()
		bob = RecordCallbackAgent()
		episode = alice.start_episode(bob)
		self.assertEqual(bob.recorded_utterances, 0)
		self.assertEqual(bob.recorded_actions, 0)
		self.assertEqual(bob.recorded_leave, 0)
		self.assertEqual(len(episode.events), 0)

		utterance1 = Event(Event.Type.Utterance, datetime.now(), alice.id, 'First utterance')
		episode.add_event(utterance1)
		self.assertEqual(len(episode.events), 1)
		self.assertEqual(episode.events[0], utterance1)
		self.assertIsNotNone(utterance1.acted)
		self.assertEqual(bob.recorded_utterances, 1)
		self.assertEqual(bob.recorded_actions, 0)
		self.assertEqual(bob.recorded_leave, 0)

		utterance2 = Event(Event.Type.Utterance, datetime.now(), bob.id, 'Bob\'s answer')
		episode.add_event(utterance2)
		self.assertEqual(len(episode.events), 2)
		self.assertEqual(episode.events[0], utterance1)
		self.assertEqual(episode.events[1], utterance2)
		self.assertTrue([e.acted is not None for e in episode.events])
		self.assertEqual(bob.recorded_utterances, 1)
		self.assertEqual(bob.recorded_actions, 0)
		self.assertEqual(bob.recorded_leave, 0)

		action1 = Event(Event.Type.Action, datetime.now(), alice.id, RecordedAction())
		episode.add_event(action1)
		self.assertEqual(len(episode.events), 3)
		self.assertEqual(episode.events[0], utterance1)
		self.assertEqual(episode.events[1], utterance2)
		self.assertEqual(episode.events[2], action1)
		self.assertTrue([e.acted is not None for e in episode.events])
		self.assertEqual(bob.recorded_utterances, 1)
		self.assertEqual(bob.recorded_actions, 1)
		self.assertEqual(bob.recorded_leave, 0)

		leave_alice = Event(Event.Type.Leave, datetime.now(), alice.id, None)
		episode.add_event(leave_alice)
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
		alice = DummyAgent()
		bob = DummyAgent()
		episode = alice.start_episode(bob)

		utterance1 = Event(Event.Type.Utterance, datetime.now(), alice.id, 'First utterance')
		action1 = Event(Event.Type.Action, datetime.now(), alice.id, RecordedAction())
		utterance2 = Event(Event.Type.Utterance, datetime.now(), alice.id, 'Second utterance')

		episode.add_event(utterance1)
		episode.add_event(action1)
		episode.add_event(utterance2)

		self.assertEqual(episode.last_event(), utterance2)
		self.assertEqual(episode.last_event(Event.Type.Utterance), utterance2)
		self.assertEqual(episode.last_event(Event.Type.Action), action1)

	def test_empty_episode_save_load(self):
		alice = DummyAgent()
		bob = DummyAgent()

		episode = alice.start_episode(bob)
		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_empty_episode_save_load.pkl')
		serialize(episode.save(), test_path)
		loaded_e = Episode.load(deserialize(test_path))

		self.assertEqual(loaded_e.id, episode.id)
		self.assertEqual(loaded_e.starter_id, episode.starter_id) 
		self.assertEqual(loaded_e.invited_id, episode.invited_id)
		self.assertEqual(len(loaded_e.events), 0)

	def test_episode_save_load(self):
		alice = DummyAgent()
		bob = DummyAgent()
		episode = alice.start_episode(bob)

		utterance1 = Event(Event.Type.Utterance, datetime.now(), alice.id, 'First utterance')
		episode.add_event(utterance1)
		action1 = Event(Event.Type.Action, datetime.now(), alice.id, RecordedAction())
		episode.add_event(action1)
		leave1 = Event(Event.Type.Leave, datetime.now(), alice.id, None)
		episode.add_event(leave1)

		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_episode_save_load.pkl')
		serialize(episode.save(), test_path)
		loaded_e = Episode.load(deserialize(test_path))

		self.assertEqual(loaded_e.starter_id, episode.starter_id) 
		self.assertEqual(loaded_e.invited_id, episode.invited_id)
		self.assertEqual(len(loaded_e.events), 3)
		self.assertEqual(loaded_e.events[0], utterance1)
		self.assertEqual(loaded_e.events[1], action1)
		self.assertEqual(loaded_e.events[2], leave1)

	def test_episode_save_load_compact(self):
		alice = DummyAgent()
		bob = DummyAgent()
		episode = alice.start_episode(bob)
		alice.say("Hi!", episode)
		bob.say("Hi!", episode)
		a = RecordedAction()
		episode.events.append(Event(Event.Type.Action, datetime.now(), 'fake-agent-id', a))
		episode.events.append(Event(Event.Type.Leave, datetime.now(), 'fale-agent-id', None))

		saved_episode = episode.save()
		saved_episode_compact = episode.save(output_format='compact')

		assert Episode.load(saved_episode) == episode
		assert Episode.load(saved_episode) == Episode.load(saved_episode_compact)

	def test_equals_true(self):
		e1 = Episode('a', 'b')
		e1.events = [
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'ddd')
		]

		e2 = Episode('a', 'b')
		e2.id = e1.id
		e2.timestamp = e1.timestamp
		e2.events = [
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'ddd')
		]

		assert e1 == e2

	def test_equals_timestamp(self):
		e1 = Episode('a', 'b')
		e1.events = [
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'ddd')
		]

		e2 = Episode('a', 'b')
		e2.id = e1.id
		e2.timestamp = datetime.now()
		e2.events = [
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'ddd')
		]

	def test_equals_events(self):
		e1 = Episode('a', 'b')
		e1.events = [
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'ddd')
		]

		e2 = Episode('a', 'b')
		e2.id = e1.id
		e2.timestamp = e1.timestamp
		e2.events = [
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'AAA'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime(2019, 12, 28), 'b', 'ddd')
		]

		assert e1 != e2

class TestExtractUtterances(unittest.TestCase):

	def test_utterances_only(self):
		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
		]
		X = extract_utterances(e)
		self.assertEqual(X, ['aaa', 'bbb', 'ccc', 'ddd'])

		X = extract_utterances(e, preprocess_f=lambda x: x.upper())
		self.assertEqual(X, ['AAA', 'BBB', 'CCC', 'DDD'])

	def test_no_holes(self):
		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
			Event(Event.Type.Action, datetime.now(), 'b', None),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
		]
		X = extract_utterances(e)
		self.assertEqual(X, ['aaa', 'ccc', 'ddd'])

		X = extract_utterances(e, preprocess_f=lambda x: x.upper())
		self.assertEqual(X, ['AAA', 'CCC', 'DDD'])

	def test_keep_holes(self):
		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
			Event(Event.Type.Action, datetime.now(), 'b', None),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
		]
		X = extract_utterances(e, keep_holes=True)
		self.assertEqual(X, ['aaa', None, 'ccc', 'ddd'])

		X = extract_utterances(e, preprocess_f=lambda x: x.upper(), keep_holes=True)
		self.assertEqual(X, ['AAA', None, 'CCC', 'DDD'])

class TestExtractUtterancePairs(unittest.TestCase):

	def test_alternate(self):
		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
		]
		X, y = extract_utterance_pairs(e)
		self.assertEqual(X, ['aaa', 'bbb', 'ccc'])
		self.assertEqual(y, ['bbb', 'ccc', 'ddd'])

	def test_repeated(self):
		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'bbb'),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
		]
		X, y = extract_utterance_pairs(e)
		self.assertEqual(X, ['ccc'])
		self.assertEqual(y, ['ddd'])

		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
		]
		X, y = extract_utterance_pairs(e)
		self.assertEqual(X, ['aaa'])
		self.assertEqual(y, ['bbb'])

		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'bbb'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'ddd')
		]
		X, y = extract_utterance_pairs(e)
		self.assertEqual(X, ['aaa', 'ccc'])
		self.assertEqual(y, ['bbb', 'ddd'])

		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'bbb'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
		]
		X, y = extract_utterance_pairs(e)
		self.assertEqual(X, ['bbb'])
		self.assertEqual(y, ['ccc'])

	def test_preprocess_f(self):
		e = Episode('a', 'b')
		e.events = [
			Event(Event.Type.Utterance, datetime.now(), 'a', 'AaA'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'BBB'),
			Event(Event.Type.Utterance, datetime.now(), 'a', 'Ccc'),
			Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
		]
		X, y = extract_utterance_pairs(e, lambda x: x.lower())
		self.assertEqual(X, ['aaa', 'bbb', 'ccc'])
		self.assertEqual(y, ['bbb', 'ccc', 'ddd'])
