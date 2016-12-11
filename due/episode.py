import logging
import uuid
import copy
from collections import namedtuple
from datetime import datetime

UTTERANCE_LABEL = 'utterance'

class Episode(object):
	"""
	An Episode is a sequence of Events
	"""

	def __init__(self, starter_agent, invited_agent):
		self._logger = logging.getLogger(__name__ + ".Episode")
		self._events = []
		self._starter = starter_agent
		self._invited = invited_agent
		self.id = uuid.uuid1()

	def add_utterance(self, agent, sentence):
		self._logger.info("New utterance by " + str(agent) + ": '" + sentence + "'")
		utterance = Event(UTTERANCE_LABEL, datetime.now(), agent, sentence)
		self._events.append(utterance)
		for a in self._other_agents(agent):
			self._logger.info("Notifying Agent " + str(a) + ".")
			a.utterance_callback(self)

	def save(self):
		return {
			'id': self.id,
			'agents': [self._starter.id, self._invited.id],
			'events': [e.save() for e in self._events]
		}

	def _other_agents(self, agent):
		return [self._starter] if agent == self._invited else [self._invited]

EventTuple = namedtuple('EventTuple', ['type', 'timestamp', 'agent', 'payload'])
class Event(EventTuple):
	"""
	An Event can be an Utterance or an Action. Note that only Utterances are
	currently modeled.
	"""
	def save(self):
		result = self._replace(agent=self.agent.id)
		return list(result)