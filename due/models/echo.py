"""
This module defines :class:`EchoAgent`, the simplest Agent of them all (after
:class:`due.agent.DummyAgent`).

API
===
"""
import logging

from due.agent import Agent

class EchoAgent(Agent):
	"""
	A simple Agent that only reacts to Utterances, and responds by simply echoing
	their content.
	"""
	def __init__(self, agent_id=None):
		super().__init__(agent_id)
		self._logger = logging.getLogger(__name__ + '.EchoAgent')

	def to_dict(self):
		"""See :meth:`due.agent.Agent.to_dict`"""
		return {'agent_id': self.id}

	@staticmethod
	def from_dict(data):
		"""See :meth:`due.agent.Agent.from_dict`"""
		return EchoAgent(**data)

	def learn_episodes(self, episodes):
		"""Dummy agents don't learn..."""
		self._logger.warning("Echo Agents don't learn...")

	def new_episode_callback(self, new_episode):
		"""See :meth:`due.agent.Agent.new_episode_callback`"""
		super().new_episode_callback(new_episode)
		self.say("Hi!", new_episode)

	def utterance_callback(self, episode, event):
		"""See :meth:`due.agent.Agent.utterance_callback`"""
		self.say(f"You said '{event.payload}'", episode)
