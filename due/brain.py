from abc import ABCMeta, abstractmethod

from due.util.python import dynamic_import

class Brain(metaclass=ABCMeta):
	"""
	A Brain is responsible for storing past episodes, and making good action
	predictions on present ones.

	`Brain` is an interface that is meant to be implemented with the greatest
	possible number of models.
	"""

	@abstractmethod
	def learn_episodes(self, episodes):
		"""
		Submit a list of Episodes for the `Brain` to learn. This just wraps calls
		to :meth:`due.brain.Brain.learn_episode`.

		TODO: this should not be an abstract method

		:param episodes: a list of episodes
		:type episodes: `list` of `due.episode.Episode`
		"""
		pass

	def learn_episode(self, episode):
		"""
		Submit an Episode for the Brain to learn.

		After the learning process, the Brain is supposed to provide better answers
		to the incoming Events. Whether this happens depends on the model implementing
		the Brain interface.

		:param episode: an Episode
		:type episode: :class:`due.episode.Episode`
		"""
		self.learn_episodes([episode])

	@abstractmethod
	def new_episode_callback(self, episode):
		"""
		Agents are supposed to call this method to notify their Brain instance
		of the initiation of a new Episode.

		See :meth:`due.agent.Agent.new_episode_callback`

		:param episode: the new Episode
		:type episode: :class:`due.episode.Episode`
		"""
		pass

	@abstractmethod
	def utterance_callback(self, episode):
		"""
		Agents are supposed to call this method to notify their Brain instance
		of a new utterance in an Episode they take part in.

		See :meth:`due.agent.Agent.utterance_callback`

		:param episode: the episode where the new utterance has been posted
		:type episode: :class:`due.episode.Episode`
		"""
		pass

	@abstractmethod
	def leave_callback(self, episode, agent):
		"""
		Agents are supposed to call this method to notify their Brain instance
		when an Agent is leaving an Episode.

		See :meth:`due.agent.Agent.leave_callback`

		:param episode: the Episode where the Agent is leaving
		:type episode: :class:`due.episode.Episode`
		:param agent: the Agent who is leaving
		:type agent: :class:`due.agent.Agent`
		"""
		pass

	@abstractmethod
	def save(self):
		"""
		Saves the Brain to a serializable object that can be reloaded with
		:meth:`due.brain.Brain.load`.
		
		A saved brain must be a dictionary containing the following items:

		* `version`: version of the class who saved the brain (often `due.__version__`)
		* `class`: absolute import name of the Brain class (eg. `due.models.vector_similarity.TfIdfCosineBrain`)
		* `data`: saved brain data. Will be passed to the Brain constructor's `_data` parameter

		:return: a serializable representation of `self`
		:rtype: `dict`
		"""
		pass

	@staticmethod
	def load(saved_brain):
		"""
		Loads an object representing a Brain that was produced by
		:meth:`due.brain.Brain.save`. 

		:param saved_brain: the saved Brain
		:type saved_brain: `dict`
		"""
		class_ = dynamic_import(saved_brain['class'])
		return class_(_data=saved_brain['data'])

class CosineBrain():

	def __init__(self):
		raise NotImplementedError("CosineBrain is deprecated. Use VectorSimilarityBrain in due.models.vector_similarity instead.")
