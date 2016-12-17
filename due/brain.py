from abc import ABCMeta, abstractmethod

import logging


class Brain(metaclass=ABCMeta):
	"""
	A Brain is responsible for storing past episodes, and making good action
	predictions on present ones.
	"""

	def __init__(self, agent):
		self._agent = agent

	@abstractmethod
	def learn_episodes(self, episodes):
		pass

	def learn_episode(self, episode):
		self.learn_episodes([episode])

	@abstractmethod
	def new_episode_callback(self, episode):
		pass

	@abstractmethod
	def utterance_callback(self, episode):
		pass

	@abstractmethod
	def save(self):
		pass

	@staticmethod
	@abstractmethod
	def load(self, saved_brain, agent):
		pass

class CosineBrain(Brain):
	"""A baseline Brain model that just uses a vetor similarity measure to pick
	appropriate answers to incoming utterances.
	"""
	def __init__(self, agent):
		self._logger = logging.getLogger(__name__ + ".CosineBrain")
		super().__init__(agent)
		self._active_episodes = {}
		self._past_episodes = []

	def learn_episodes(self, episodes):
		self._past_episodes.extend(episodes)

	def new_episode_callback(self, episode):
		self._active_episodes[episode.id] = episode

	def utterance_callback(self, episode):
		episode.add_utterance(self._agent, "Not implemented yet: this is just a default answer...")

	def save(self):
		return {
			'past_episodes': [e.save() for e in self._past_episodes]
		}

	@staticmethod
	def load(saved_brain, agent):
		result = CosineBrain(agent)
		result._past_episodes = saved_brain['past_episodes']
		return result