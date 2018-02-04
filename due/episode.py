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

	def __init__(self, starter_agent, invited_agent):
		self._logger = logging.getLogger(__name__ + ".Episode")
		self._starter = starter_agent
		self._invited = invited_agent
		self.id = uuid.uuid1()
		self.events = []

	def add_event(self, agent, event):
		self._logger.info("New %s event by %s: '%s'" % (event.type.name, agent, event.payload))
		self.events.append(event)
		event.mark_acted()
		for a in self._other_agents(agent):
			self._logger.debug("Notifying Agent %s." % a)
			a.event_callback(event, self)
			# TODO: comlete (implement event_callback in agent etc.)

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
			'agents': [str(self._starter.id), str(self._invited.id)],
			'events': [e.save() for e in self.events]
		}

	def _other_agents(self, agent):
		return [self._starter] if agent == self._invited else [self._invited]

	#
	# Deprecated Event handling
	#

	def add_utterance(self, agent, sentence):
		self._logger.warn('deprecated: please update your code to use Episode.add_event() instead.')
		self._logger.info("New utterance by %s: '%s'" % (agent, sentence))
		utterance = Event(Event.Type.Utterance, datetime.now(), agent, sentence)
		self.events.append(utterance)
		for a in self._other_agents(agent):
			self._logger.debug("Notifying Agent %s." % a)
			a.utterance_callback(self)

	def add_action(self, agent, action):
		self._logger.warn('deprecated: please update your code to use Episode.add_event() instead.')
		self._logger.info("New action by %s: '%s'" % (agent, action))
		action_event = Event(Event.Type.Action, datetime.now(), agent, action)
		self.events.append(action_event)
		for a in self._other_agents(agent):
			self._logger.debug("Notifying Agent %s." % a)
			a.action_callback(self)

	def leave(self, agent):
		self._logger.info("Agent %s left." % agent)
		event = Event(Event.Type.Leave, datetime.now(), agent, None)
		for a in self._other_agents(agent):
			self._logger.debug("Notifying Agent %s." % a)
			a.leave_callback(self, agent)