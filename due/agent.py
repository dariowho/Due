"""
Due is a learning, modular, action-oriented dialogue agent.
"""
import uuid
from abc import ABCMeta, abstractmethod

import logging

from due.episode import Episode

class Agent(metaclass=ABCMeta):
	"""
	An Agent is any of the participants in an episode. It only models an unique
	identity through its ID. Optionally, a human-friendly name can be provided,
	which should not be taken into account in Machine Learning processes.
	"""

	def __init__(self, id=None, name=None):
		self.id = id if id is not None else uuid.uuid1()
		self.name = name

	def save(self):
		return {'id': self.id}

	@staticmethod
	def load(exported_agent):
		return Agent(exported_agent['id'])

	@abstractmethod
	def new_episode_callback(self, new_episode):
		"""
		Another Agent will use this method to notify its intention to start a
		new conversation (an Episode).

		:param new_episode: the new Episode that the other Agent has created
		:type new_episode: :class:`due.episode.Episode`
		"""
		pass

	@abstractmethod
	def utterance_callback(self, episode):
		"""
		When one of the Agents in the episode adds an utterance, the Episode
		will call this method on the other Agents to notify the change. 
		"""
		pass

	def __str__(self):
		name = self.name if self.name is not None else self.id
		return "<Agent: " + name + ">"

class HumanAgent(Agent):
	"""
	A Human Agent is an Agent that uses a Human brain, typically sitting behind
	a keyboard, to make sense of :class:`Episode`\ s and produce new
	:class:`.Event`\ s.
	"""
	def __init__(self, id=None, name=None):
		super().__init__(id, name)
		self._active_episodes = {}
		self._logger = logging.getLogger(__name__ + '.HumanAgent')

	def start_episode(self, other):
		"""
		Starts a conversational Episode with another Agent.

		:param other_agent: The Agent you are including in the conversation.
		:type other_agent: :class:`due.agent.Agent`

		:rtype: :class:`due.episode.Episode`
		"""
		result = Episode(self, other)
		other.new_episode_callback(result)
		return result

	def say(self, sentence, episode):
		"""
		Adds the given sentence to the given Episode

		:param sentence: A sentence
		:type sentence: :class:`str`
		:param episode: An Episode
		:type episode: :class:`due.episode.Episode`
		"""
		episode.add_utterance(self, sentence)

	def new_episode_callback(self, new_episode):
		self._logger.info("New episode callback: " + str(new_episode))
		self._active_episodes[new_episode.id] = new_episode

	def utterance_callback(self, episode):
		self._logger.info("Utterance received.")

class Due(Agent):
	"""
	Main entry point for Due. Should be instantiated with a Brain
	"""

	DEFAULT_UUID = uuid.UUID('423cc038-bfe0-11e6-84d6-a434d9562d81')

	def __init__(self, brain=None):
		super().__init__(Due.DEFAULT_UUID, "Due")
		self._active_episodes = {}
		self._brain = brain
		self._logger = logging.getLogger(__name__ + ".Due")

	def new_episode_callback(self, new_episode):
		self._logger.info("Got invited to a new episode.")
		self._active_episodes[new_episode.id] = new_episode

	def utterance_callback(self, episode):
		self._logger.info("Received utterance")
		self._logger.info("This is where I should do something with the Brain...")
		episode.add_utterance(self, "No brain yet: this is just a default answer...")
