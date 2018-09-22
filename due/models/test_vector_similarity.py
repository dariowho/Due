import unittest
from datetime import datetime
import tempfile
import os

from numpy.testing import assert_array_equal

from due.brain import Brain
from due.episode import Episode
from due.event import Event
from due.persistence import serialize, deserialize
from due.models.vector_similarity import TfIdfCosineBrain

class TestTfIdfCosineBrain(unittest.TestCase):

	def test_save_load(self):

		brain = TfIdfCosineBrain()
		brain.learn_episodes(_get_train_episodes())
		saved_brain = brain.save()

		with tempfile.TemporaryDirectory() as temp_dir:
			path = os.path.join(temp_dir, 'serialized_vector_brain.due')
			serialize(saved_brain, path)
			loaded_brain = Brain.load(deserialize(path))

		self.assertEqual(brain.parameters, loaded_brain.parameters)
		self.assertEqual(brain._normalized_past_utterances, loaded_brain._normalized_past_utterances)
		self.assertEqual([e.save() for e in loaded_brain._past_episodes], [e.save() for e in brain._past_episodes])
		expected_utterance = brain._process_utterance('aaa bbb ccc mario')
		loaded_utterance = loaded_brain._process_utterance('aaa bbb ccc mario')
		self.assertEqual((brain._vectorizer.transform([expected_utterance]) != loaded_brain._vectorizer.transform([loaded_utterance])).nnz, 0)
		self.assertEqual((brain._vectorized_past_utterances != loaded_brain._vectorized_past_utterances).nnz, 0)

		self.assertEqual(brain.utterance_callback(_get_test_episode())[0].payload, loaded_brain.utterance_callback(_get_test_episode())[0].payload)


	def test_utterance_callback(self):
		brain = TfIdfCosineBrain()
		brain.learn_episodes(_get_train_episodes())
		saved_brain = brain.save()
		result = brain.utterance_callback(_get_test_episode())
		self.assertEqual(result[0].payload, 'bbb')

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

	e = Episode('a', 'b')
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