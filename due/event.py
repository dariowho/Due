"""
Events represent changes in the state of an Episode. Currently, three event
types are supported:

* Utterances: an Agent said something
* Leave: an Agent left the Episode
* Action: an Agent performed an :class:`due.action.Action` in a episode

Episodes themselves are defined as sequences of Events.

API
===
"""
from collections import namedtuple
from enum import Enum
from datetime import datetime
import logging

from due.action import Action
from due.util.time import convert_datetime

EventTuple = namedtuple('EventTuple', ['type', 'timestamp', 'agent', 'payload'])

class Event(EventTuple):
	"""
	An Event is anything that can happen in an Episode. It can be an Utterance,
	an Action, or a Leave event.
	"""

	class Type(Enum):
		"""
		Enumerates the three Event types:

		* `Event.Type.Utterance`
		* `Event.Type.Leave`
		* `Event.Type.Action`
		"""
		Utterance = "utterance"
		Leave = "leave"
		Action = "action"

	def __init__(self, *args, **kwargs):
		super().__init__()
		# EventTuple.__init__(self, *args, **kwargs)
		self._logger = logging.getLogger(__name__)
		if not isinstance(self.timestamp, datetime):
			raise ValueError('timestamp value is not a `datetime` instance: ' \
							     'please update your code to avoid unexpected errors.')
		if self.agent and not isinstance(self.agent, str):
			raise ValueError('`agent` value is not a `str` object. Please provide a ' \
				             'string ID to ensure correct serialization.')
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

		:return: a saved Event
		:rtype: `list`
		"""
		result = self._replace(
			type=self.type.value,
			timestamp=self.timestamp.isoformat()
		)
		if self.type == Event.Type.Action:
			result = result._replace(payload=self.payload.save())

		return dict(result._asdict())

	@staticmethod
	def load(saved):
		"""
		Load an Event that was saved with :func:`due.event.Event.save`

		:param saved: the saved Event
		:type saved: `list`
		"""
		_action = Event.Type.Action.value
		return Event(
			Event.Type(saved['type']),
			convert_datetime(saved['timestamp']),
			saved['agent'],
			Action.load(saved['payload']) if saved['type'] == _action else saved['payload']
		)

	def clone(self):
		"""
		Clone the Event into a new object.

		:return: a new Event object that is a clone of `self`
		:rtype: :class:`due.event.Event`
		"""
		return Event(*list(self))
