import logging

from due.agent import Agent
from due.episode import LiveEpisode
from due.event import Event

class DummyAgent(Agent):
	"""
	A Dummy Agent is an Agent that simply logs new Episodes and Events,
	expecting the interaction to be commanded externally.
	"""
	def __init__(self, agent_id=None):
		super().__init__(agent_id)
		self._active_episodes = {}
		self._logger = logging.getLogger(__name__ + '.DummyAgent')

	def save(self):
		"""See :meth:`due.agent.Agent.save`"""
		return {'agent_id': self.id}

	@staticmethod
	def load(saved_agent):
		"""See :meth:`due.agent.Agent.load`"""
		return DummyAgent(**saved_agent)

	def learn_episodes(self, new_episodes):
		"""Dummy agents don't learn..."""
		pass
	
	def start_episode(self, other):
		"""
		Create a new :class:`due.episode.Episode` to engage another Agent in a
		new conversation.

		:param other_agent: The Agent you are inviting to the conversation.
		:type other_agent: :class:`due.agent.Agent`
		:return: a new Episode object
		:rtype: :class:`due.episode.LiveEpisode`
		"""
		result = LiveEpisode(self, other)
		other.new_episode_callback(result)
		return result

	def new_episode_callback(self, new_episode):
		"""See :meth:`due.agent.Agent.new_episode_callback`"""
		self._logger.info("New episode callback: %s", str(new_episode))
		self._active_episodes[new_episode.id] = new_episode

	def utterance_callback(self, episode):
		"""See :meth:`due.agent.Agent.utterance_callback`"""
		self._logger.debug("Utterance received.")

	def action_callback(self, episode):
		"""See :meth:`due.agent.Agent.action_callback`"""
		self._logger.debug("Action received.")

	def leave_callback(self, episode):
		"""See :meth:`due.agent.Agent.leave_callback`"""
		agent = episode.last_event(Event.Type.Leave).agent
		self._logger.debug("Agent %s left the episode.", agent)
