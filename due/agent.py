"""
Due is a learning, modular, action-oriented dialogue agent.
"""
import uuid
from abc import ABCMeta, abstractmethod

import logging

from due.episode import Episode
from due.brain import Brain
from due.brain import CosineBrain

class Agent(metaclass=ABCMeta):
	"""
	An Agent is any of the participants in an episode. It only models an unique
	identity through its ID, and handles the communication on the channel level.
	Optionally, a human-friendly name can be provided, which should not be taken
	into account in Machine Learning processes.
	"""

	def __init__(self, id=None, name=None):
		self.id = id if id is not None else uuid.uuid1()
		self.name = name

	@abstractmethod
	def save(self):
		pass

	@staticmethod
	@abstractmethod
	def load(exported_agent):
		pass

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

	@abstractmethod
	def action_callback(self, episode):
		"""
		When one of the agents in the episode performs an action, the Episode
		will call this method on the other Agents to notify the change.
		"""
		pass

	@abstractmethod
	def leave_callback(self, episode, agent):
		"""
		When one of the Agents in the episode leaves, the Episode will call this
		method on the other participants to notify the change.
		"""
		pass

	def say(self, sentence, episode):
		"""
		Say the given sentence in the given Episode. By default, it just updates
		the Episode, but you may want to extend the implementation with some
		output operation (eg. printing on screen, broadcasing to a jabber chat...)
		
		:param sentence: A sentence
		:type sentence: :class:`str`
		:param episode: An Episode
		:type episode: :class:`due.episode.Episode`
		"""
		episode.add_utterance(self, sentence)

	def do(self, action, episode):
		"""
		Performs the given action in the given Episode

		:param action: An Action
		:type action: :class:`due.action.Action`
		"""
		action.run()
		episode.add_action(self, action)

	def leave(self, episode):
		"""
		Leaves the given episode.

		:param episode: One of the Agent's active episodes
		:type episode: :class:`due.episode.Episode`
		"""
		episode.leave(self)

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

	def save(self):
		return {'id': self.id, 'name': self.name}

	@staticmethod
	def load(saved_agent):
		return HumanAgent(**saved_agent)

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

	def new_episode_callback(self, new_episode):
		self._logger.info("New episode callback: " + str(new_episode))
		self._active_episodes[new_episode.id] = new_episode

	def utterance_callback(self, episode):
		self._logger.debug("Utterance received.")

	def action_callback(self, episode):
		self._logger.debug("Action received.")

	def leave_callback(self, episode, agent):
		self._logger.debug("Agent %s left the episode." % agent)

DEFAULT_DUE_UUID = uuid.UUID('423cc038-bfe0-11e6-84d6-a434d9562d81')

class Due(Agent):
	"""
	Main entry point for Due. Should be instantiated with a Brain
	"""

	def __init__(self, id=DEFAULT_DUE_UUID, brain=None):
		Agent.__init__(self, id, "Due")
		
		if isinstance(brain, dict):
			brain = Brain.load(brain, self)

		brain = brain if brain is not None else CosineBrain(self)
		self._brain = brain
		self._logger = logging.getLogger(__name__ + ".Due")

	def learn_episodes(self, episodes):	
		self._logger.info("Learning some episodes.")
		self._brain.learn_episodes(episodes)

	def new_episode_callback(self, episode):
		self._logger.info("Got invited to a new episode.")
		self._brain.new_episode_callback(episode)

	def utterance_callback(self, episode):
		self._logger.debug("Received utterance")
		self._brain.utterance_callback(episode)

	def action_callback(self, episode):
		self._logger.debug("Received action")

	def leave_callback(self, episode, agent):
		self._logger.debug("Agent %s left the episode." % agent)
		self._brain.leave_callback(episode, agent)

	@staticmethod
	def load(saved_agent):
		return Due(saved_agent['id'], saved_agent['brain'])

	def save(self):
		return {
			'id': self.id,
			'name': self.name,
			'brain': self._brain.save()
		}