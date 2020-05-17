import unittest
from datetime import datetime
import tempfile
import os

from due.agent import Agent
from due.episode import Episode
from due.event import Event
from due.persistence import serialize, deserialize
from due.models.tfidf import TfIdfAgent
from due.agent import DummyAgent

class TestTfIdfAgent(unittest.TestCase):

	def test_save_load(self):

		agent = TfIdfAgent()
		agent.learn_episodes(_get_train_episodes())
		saved_agent = agent.save()

		with tempfile.TemporaryDirectory() as temp_dir:
			path = os.path.join(temp_dir, 'serialized_tfidf_agent.due')
			serialize(saved_agent, path)
			loaded_agent = Agent.load(deserialize(path))

		assert agent.parameters == loaded_agent.parameters
		assert agent._normalized_past_utterances == loaded_agent._normalized_past_utterances
		assert [e.save() for e in loaded_agent._past_episodes] == [e.save() for e in agent._past_episodes]
		expected_utterance = agent._process_utterance('aaa bbb ccc mario')
		loaded_utterance = loaded_agent._process_utterance('aaa bbb ccc mario')
		assert (agent._vectorizer.transform([expected_utterance]) != loaded_agent._vectorizer.transform([loaded_utterance])).nnz == 0
		assert (agent._vectorized_past_utterances != loaded_agent._vectorized_past_utterances).nnz == 0

		episode = _get_test_episode()
		assert agent.utterance_callback(episode, episode.events[-1])[0].payload, loaded_agent.utterance_callback(_get_test_episode())[0].payload


	def test_utterance_callback(self):
		agent = TfIdfAgent()
		agent.learn_episodes(_get_train_episodes())
		episode = _get_test_episode()
		result = agent.utterance_callback(episode, episode.events[-1])
		self.assertEqual(result[0].payload, 'bbb')

	def test_tfidf_agent(self):
		cb = TfIdfAgent()

		# Learn sample episode
		sample_episode, alice, bob = _sample_episode()
		cb.learn_episodes([sample_episode])

		# Predict answer
		e2 = alice.start_episode(bob)
		alice.say("Hi!", e2)
		answer_events = cb.utterance_callback(e2, e2.events[-1])
		self.assertEqual(len(answer_events), 1)
		self.assertEqual(answer_events[0].payload, 'Hello')
		
	def test_agent_load(self):
		sample_episode, alice, bob = _sample_episode()
		cb = TfIdfAgent()
		cb.learn_episodes([sample_episode])
		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_agent_load.pkl')
		serialize(cb.save(), test_path)

		loaded_cb = Agent.load(deserialize(test_path))

		self.assertIsInstance(loaded_cb, TfIdfAgent)

		e2 = alice.start_episode(bob)
		alice.say("Hi!", e2)
		answer_events = loaded_cb.utterance_callback(e2, e2.events[-1])
		self.assertEqual(len(answer_events), 1)
		self.assertEqual(answer_events[0].payload, 'Hello')

def _get_train_episodes():
	result = []

	e = Episode('a', 'b')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
		Event(Event.Type.Utterance, datetime.now(), 'b', 'bbb'),
		Event(Event.Type.Utterance, datetime.now(), 'a', 'ccc'),
		Event(Event.Type.Utterance, datetime.now(), 'b', 'ddd')
	]
	result.append(e)

	e = Episode('1', '2')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), '1', '111'),
		Event(Event.Type.Utterance, datetime.now(), '2', '222'),
		Event(Event.Type.Utterance, datetime.now(), '1', '333'),
		Event(Event.Type.Utterance, datetime.now(), '2', '444')
	]
	result.append(e)

	return result

def _get_test_episode():
	e = Episode('a', 'b')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
	]
	return e

def _sample_episode():
	alice = DummyAgent()
	bob = DummyAgent()
	result = alice.start_episode(bob)
	alice.say("Hi!", result)
	bob.say("Hello", result)
	alice.say("How are you?", result)
	bob.say("Good thanks, and you?", result)
	alice.say("All good", result)

	return result, alice, bob
