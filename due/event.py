from collections import namedtuple
from enum import Enum
from datetime import datetime
import logging

EventTuple = namedtuple('EventTuple', ['type', 'timestamp', 'agent', 'payload'])

class Event(EventTuple):
	"""
	An Event can be an Utterance or an Action. Note that only Utterances are
	currently modeled.
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
		self.acted = None

	def mark_acted(self, timestamp=None):
		"""
		Mark the Event as acted storing the timestamp of the moment the event 
		was acted.

		If no timestamp is given, the current timestamp is used.

		:param timestamp: timestamp of the moment the event was acted
		:type timestamp: `datetime`
		"""
		self.acted = timestamp if timestamp else datetime.now()

	def save(self):
		"""
		Export the Event to a serializable `list`.
		"""
		result = self._replace(type=self.type.value, timestamp=self.timestamp.isoformat(), agent=self.agent.id)
		if self.type == Event.Type.Action:
			result = result._replace(payload=full_class_name(self.payload))
		return list(result)

	def clone(self):
		return Event(*list(self))