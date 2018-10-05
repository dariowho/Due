import logging
import uuid

from due.agent import Agent
from due.brain import Brain
from due.event import Event

from due.models.vector_similarity import TfIdfCosineBrain

DEFAULT_DUE_UUID = uuid.UUID('423cc038-bfe0-11e6-84d6-a434d9562d81')

class Due(Agent):
	"""
	Due is a conversational agent that makes use of a :class:`due.brain.Brain`
	object to understand events and produce new ones.

	Due is a subclass of :class:`due.agent.Agent`.
	"""

	def __init__(self, agent_id=DEFAULT_DUE_UUID, brain=None):

		Agent.__init__(self, agent_id, "Due")

		if isinstance(brain, dict):
			brain = Brain.load(brain)

		brain = brain if brain is not None else TfIdfCosineBrain()
		self._brain = brain
		self._logger = logging.getLogger(__name__ + ".Due")

	def learn_episodes(self, episodes):
		"""
		Trains the Natural Language and Generation (Brain) model with the
		given sequence of Episodes.

		:param episodes: a list of Episodes
		:type episodes: `list` of `due.episode.Episode`
		"""
		self._logger.info("Learning some episodes.")
		self._brain.learn_episodes(episodes)

	def new_episode_callback(self, new_episode):
		"""See :meth:`due.agent.Agent.new_episode_callback`"""
		self._logger.info("Got invited to a new episode.")
		self._brain.new_episode_callback(new_episode)

	def utterance_callback(self, episode):
		"""See :meth:`due.agent.Agent.utterance_callback`"""
		self._logger.debug("Received utterance")
		answers = self._brain.utterance_callback(episode)
		self.act_events(answers, episode)

	def action_callback(self, episode):
		"""See :meth:`due.agent.Agent.action_callback`"""
		self._logger.debug("Received action")

	def leave_callback(self, episode):
		"""See :meth:`due.agent.Agent.leave_callback`"""
		agent = episode.last_event(Event.Type.Leave).agent
		self._logger.debug("Agent %s left the episode.", agent)
		self._brain.leave_callback(episode, agent)

	@staticmethod
	def load(saved_agent):
		"""See :meth:`due.agent.Agent.load`"""
		return Due(saved_agent['id'], saved_agent['brain'])

	def save(self):
		"""See :meth:`due.agent.Agent.save`"""
		return {
			'id': self.id,
			'name': self.name,
			'brain': self._brain.save()
		}
