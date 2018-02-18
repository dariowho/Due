from due.util import full_class_name
from due.event import Event

import logging
import uuid
import copy
from datetime import datetime

UTTERANCE_LABEL = 'utterance'

class Episode(object):
	"""
	An Episode is a sequence of Events
	"""

	def __init__(self, starter_agent_id, invited_agent_id):
		self._logger = logging.getLogger(__name__ + ".Episode")
		self._starter_id = starter_agent_id
		self._invited_id = invited_agent_id
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
		return {
			'id': self.id,
			'starter_agent': str(self._starter_id),
			'invited_agents': [str(self._invited_id)],
			'events': [e.save() for e in self.events]
		}

	@staticmethod
	def load(other):
		result = Episode(other['starter_agent'], other['invited_agents'][0])
		result.id = other['id']
		result.events = [Event.load(e) for e in other['events']]
		return result

class LiveEpisode(Episode):

	def __init__(self, starter_agent, invited_agent):
		super().__init__(starter_agent.id, invited_agent.id)
		self._starter = starter_agent
		self._invited = invited_agent

	def add_event(self, agent, event):
		self._logger.info("New %s event by %s: '%s'" % (event.type.name, agent, event.payload))
		self.events.append(event)
		event.mark_acted()
		for a in self._other_agents(agent):
			self._logger.debug("Notifying Agent %s." % a)
			a.event_callback(event, self)

	def _other_agents(self, agent):
		return [self._starter] if agent == self._invited else [self._invited]