from due.util.python import full_class_name

import logging
import uuid
import copy

UTTERANCE_LABEL = 'utterance'

class Episode(object):
	"""
	An Episode is a sequence of Events issued by Agents
	"""

	def __init__(self, starter_agent_id, invited_agent_id):
		self._logger = logging.getLogger(__name__ + ".Episode")
		self.starter_id = starter_agent_id
		self.invited_id = invited_agent_id
		self.id = uuid.uuid1()
		self.events = []

	def last_event(self, event_type=None):
		"""
		Returns the last event in the Episode. Optionally, events can be filtered
		by type.

		:param event_type: an event type, or a collection of types
		:type event_type: :class:`Event.Type` or list of :class:`Event.Type`
		"""
		self._logger.info(event_type)
		self._logger.info(type(event_type))
		self._logger.info(isinstance(event_type, Event.Type))
		event_type = event_type if not isinstance(event_type, Event.Type) else (event_type,)
		self._logger.info(event_type)
		self._logger.info(type(event_type))
		for e in reversed(self.events):
			if event_type is None or e.type in event_type:
				return e
		return None

	def save(self):
		"""
		Save the Episode to a serializable object, that can be loaded with
		:meth:`due.episode.Episode.load`.

		:return: a serializable representation of `self`
		:rtype: `dict`
		"""
		return {
			'id': self.id,
			'starter_agent': str(self.starter_id),
			'invited_agents': [str(self.invited_id)],
			'events': [e.save() for e in self.events]
		}

	@staticmethod
	def load(other):
		"""
		Loads an Episode as it was saved by :meth:`due.episode.Episode.save`.

		:param other: the episode to be loaded
		:type other: `dict`
		:return: an Episode object representing `other`
		:rtype: :class:`due.episode.Episode`
		"""
		result = Episode(other['starter_agent'], other['invited_agents'][0])
		result.id = other['id']
		result.events = [Event.load(e) for e in other['events']]
		return result

class LiveEpisode(Episode):
	"""
	A LiveEpisode is an Episode that is currently under way. That is, new Events
	can be acted in it.

	:param starter_agent: the Agent which started the Episode
	:type starter_agent: :class:`due.agent.Agent`
	:param invited_agent: the agent invited to the Episode
	:type invited_agent: :class:`due.agent.Agent`
	"""
	def __init__(self, starter_agent, invited_agent):
		super().__init__(starter_agent.id, invited_agent.id)
		self.starter = starter_agent
		self.invited = invited_agent

	def add_event(self, agent, event):
		"""
		Adds an Event to the LiveEpisode.

		:param agent: the agent which acted the Event
		:type agent: :class:`due.agent.Agent`
		:param event: the event that was acted by the Agent
		:type event: :class:`due.event.Event`
		"""
		self._logger.info("New %s event by %s: '%s'" % (event.type.name, agent, event.payload))
		self.events.append(event)
		event.mark_acted()
		for a in self._other_agents(agent):
			self._logger.debug("Notifying Agent %s." % a)
			a.event_callback(event, self)

	def _other_agents(self, agent):
		return [self.starter] if agent == self.invited else [self.invited]

# Quick fix for circular dependencies
from due.event import Event
