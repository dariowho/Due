import unittest
import tempfile
import os

from due.persistence import serialize, deserialize
from due.agent import HumanAgent
from due.brain import *
from due.models.vector_similarity import TfIdfCosineBrain

class TestBrain(unittest.TestCase):

	def setUp(self):
		self.alice = HumanAgent(name="Alice")
		self.bob = HumanAgent(name="Bob")
		self.sample_episode = self.alice.start_episode(self.bob)
		self.alice.say("Hi!", self.sample_episode)
		self.bob.say("Hello", self.sample_episode)
		self.alice.say("How are you?", self.sample_episode)
		self.bob.say("Good thanks, and you?", self.sample_episode)
		self.alice.say("All good", self.sample_episode)

	def test_tfidfcosine_brain(self):
		cb = TfIdfCosineBrain()

		# Learn sample episode
		cb.learn_episodes([self.sample_episode])

		# Predict answer
		e2 = self.alice.start_episode(self.bob)
		self.alice.say("Hi!", e2)
		answer_events = cb.utterance_callback(e2)
		self.assertEqual(len(answer_events), 1)
		self.assertEqual(answer_events[0].payload, 'Hello')
		
	def test_brain_load(self):
		cb = TfIdfCosineBrain()
		cb.learn_episodes([self.sample_episode])
		test_dir = tempfile.mkdtemp()
		test_path = os.path.join(test_dir, 'test_brain_load.pkl')
		saved_cb = serialize(cb.save(), test_path)

		loaded_cb = Brain.load(deserialize(test_path))

		self.assertIsInstance(loaded_cb, TfIdfCosineBrain)

		e2 = self.alice.start_episode(self.bob)
		self.alice.say("Hi!", e2)
		answer_events = loaded_cb.utterance_callback(e2)
		self.assertEqual(len(answer_events), 1)
		self.assertEqual(answer_events[0].payload, 'Hello')