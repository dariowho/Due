from collections import namedtuple
from enum import Enum
from datetime import datetime
import logging
from dateutil.parser import parse as dateutil_parse

from due.action import Action
from due.util import full_class_name
from . import agent

EventTuple = namedtuple('EventTuple', ['type', 'timestamp', 'agent', 'payload'])

class Event(EventTuple):
	"""
	An Event can be an Utterance, an Action, or a Leave event.
	"""

	class Type(Enum):
		Utterance = "utterance"
		Leave = "leave"
		Action = "action"

	def __init__(self, *args, **kwargs):
		super().__init__()
		# EventTuple.__init__(self, *args, **kwargs)
		self._logger = logging.getLogger(__name__)
		if not isinstance(self.timestamp, datetime):
			self._logger.warn('timestamp value is not a `datetime` instance: please update your code to avoid unexpected errors.')
		if isinstance(self.agent, agent.Agent):
			self._logger.warn('agent value is an `Agent` object: please provide a string ID instead to ensure correct serialization.')
		self.acted = None

	def mark_acted(self, timestamp=None):
		"""
		Mark the Event as acted storing the timestamp of the moment the event 
		was acted. An Event is acted when it's issued in an episode.

		If no timestamp is given, the current timestamp is used.

		:param timestamp: timestamp of the moment the event was acted
		:type timestamp: `datetime`
		"""
		self.acted = timestamp if timestamp else datetime.now()

	def save(self):
		"""
		Export the Event to a serializable `list`.
		"""
		result = self._replace(type=self.type.value,
			                   timestamp=self.timestamp.isoformat())
		if self.type == Event.Type.Action:
			result = result._replace(payload=self.payload.save())
		return list(result)

	@staticmethod
	def load(saved):
		"""
		Load a saved Event
		"""
		loaded_type = Event.Type(saved[0])
		loaded_payload = Action.load(saved[3]) if loaded_type == Event.Type.Action else saved[3]
		return Event(loaded_type,
			         dateutil_parse(saved[1]),
			         saved[2],
			         loaded_payload)

	def clone(self):
		return Event(*list(self))