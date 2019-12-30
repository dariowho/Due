import unittest
from datetime import datetime
import tempfile
import os

import torch
import pytest

from due.agent import Agent
from due.episode import Episode
from due.event import Event
from due.persistence import serialize, deserialize
from due.models.seq2seq import EncoderDecoderAgent

pytest.skip("skipping Pytorch tests: that code will be moved", allow_module_level=True)

class TestEncoderDecoderAgent(unittest.TestCase):

	def test_save_load(self):
		agent = EncoderDecoderAgent({
			'batch_size': 1,
			'hidden_size': 16,
		}, _get_train_episodes(), random_embedding_init=True)
		saved_agent = agent.save()

		with tempfile.TemporaryDirectory() as temp_dir:
			path = os.path.join(temp_dir, 'serialized_encdec_agent.due')
			serialize(saved_agent, path)
			loaded_agent = Agent.load(deserialize(path))

		self.assertEqual(agent.X, loaded_agent.X)
		self.assertEqual(agent.y, loaded_agent.y)
		self.assertEqual(agent.vocabulary.save(), loaded_agent.vocabulary.save())
		self.assertEqual(agent.parameters, loaded_agent.parameters)
		assert torch.all(torch.eq(agent.embedding_matrix, loaded_agent.embedding_matrix))

		loaded_encoder_state = loaded_agent.encoder.state_dict()
		for k, v in agent.encoder.state_dict().items():
			assert torch.all(torch.eq(v, loaded_encoder_state[k]))

		loaded_decoder_state = loaded_agent.decoder.state_dict()
		for k, v in agent.decoder.state_dict().items():
			assert torch.all(torch.eq(v, loaded_decoder_state[k]))

	def test_reset_with_parameters(self):
		agent = EncoderDecoderAgent({
			'batch_size': 1,
			'hidden_size': 16,
		}, _get_train_episodes(), random_embedding_init=True)

		agent_new = agent.reset_with_parameters({'hidden_size': 8})
		self.assertEqual(agent.X, agent_new.X)
		self.assertEqual(agent.y, agent_new.y)
		self.assertEqual(agent.vocabulary.save(), agent_new.vocabulary.save())
		assert torch.all(torch.eq(agent.embedding_matrix, agent_new.embedding_matrix))
		self.assertEqual(agent_new.parameters, {**agent_new.parameters, **{'hidden_size': 8}})
		self.assertEqual(agent.encoder.gru.hidden_size, 16)
		self.assertEqual(agent_new.encoder.gru.hidden_size, 8)
		self.assertEqual(agent.decoder.gru.hidden_size, 16)
		self.assertEqual(agent_new.decoder.gru.hidden_size, 8)

	def test_epoch_predict(self):
		agent = EncoderDecoderAgent({
			'batch_size': 2,
			'hidden_size': 16,
		}, _get_train_episodes(), random_embedding_init=True)
		agent.epoch()
		agent.predict('just an utterance')

	def test_long_utterances(self):
		agent = EncoderDecoderAgent({
			'batch_size': 2,
			'hidden_size': 16,
			'max_sentence_length': 2
		}, _get_train_episodes(), random_embedding_init=True)
		agent.epoch()
		agent.predict('just an utterance')

def _get_train_episodes():
	result = []

	e = Episode('a', 'b')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), 'a', 'this is the first utterance'),
		Event(Event.Type.Utterance, datetime.now(), 'b', 'this is the first answer'),
		Event(Event.Type.Utterance, datetime.now(), 'a', 'this one is the second utterance'),
		Event(Event.Type.Utterance, datetime.now(), 'b', 'this is the second answer')
	]
	result.append(e)

	e = Episode('1', '2')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), '1', 'one one one'),
		Event(Event.Type.Utterance, datetime.now(), '2', 'two two two'),
		Event(Event.Type.Utterance, datetime.now(), '1', 'three three three'),
		Event(Event.Type.Utterance, datetime.now(), '2', 'four four four')
	]
	result.append(e)

	return result

def _get_test_episode():
	e = Episode('a', 'b')
	e.events = [
		Event(Event.Type.Utterance, datetime.now(), 'a', 'aaa'),
	]
	return e
