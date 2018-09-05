from due.util.python import full_class_name

import logging
import uuid
import copy
from functools import lru_cache

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
	def load(saved_episode):
		"""
		Loads an Episode as it was saved by :meth:`due.episode.Episode.save`.

		:param saved_episode: the episode to be loaded
		:type saved_episode: `dict`
		:return: an Episode object representing `saved_episode`
		:rtype: :class:`due.episode.Episode`
		"""
		result = Episode(saved_episode['starter_agent'], saved_episode['invited_agents'][0])
		result.id = saved_episode['id']
		result.events = [Event.load(e) for e in saved_episode['events']]
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

def _is_utterance(event):
	return event.type == Event.Type.Utterance

def extract_utterances(episode, preprocess_f=None, keep_holes=False):
	"""
	Return all the utterances in an Episode as strings. If the `keep_holes`
	parameter is set `True`, non-utterance events will be returned as well, as
	`None` elements in the resulting list.

	:param episode: the Episode to extract utterances from
	:type episode: :class:`Episode`
	:param preprocess_f: when given, sentences will be run through this function before being returned
	:type preprocess_f: `func`
	:param keep_holes: if `True`, `None` elements will be returned in place of non-utterance events.
	:return: the list of utterances in the Episode
	:rtype: `list` of `str`
	"""
	if not preprocess_f:
		preprocess_f = lambda x: x

	result = []
	for e in episode.events:
		if e.type == Event.Type.Utterance:
			result.append(preprocess_f(e.payload))
		elif keep_holes:
			result.append(None)

	return result

def extract_utterance_pairs(episode, preprocess_f=None):
	"""
	Process Events in an Episode, extracting all the Utterance Event pairs that
	can be interpreted as one dialogue turn (ie. an Agent's utterance, and a
	different Agent's response).

	In particular, Event pairs are extracted from the Episode so that:

	* Both Events are Utterances (currently, non-utterances will raise an exception)
	* The second Event immediately follows the first
	* The two Events are acted by two different Agents

	This means that if an utterance has more than one answers, only the first
	one will be included in the result.

	If a `preprocess_f` function is specified, resulting utterances will be run
	through this function before being returned. A LRU Cache is applied to
	`preprocess_f`, as most sentences will be returned as both utterances and
	answers/

	Return two lists of the same length, so that each utterance `X_i` in the
	first list has its response `y_i` in the second.

	:param episode: an Episode
	:type episode: :class:`due.episode.Episode`
	:param preprocess_f: when given, sentences will be run through this function before being returned
	:type preprocess_f: `func`
	:return: a list of utterances and the list of their answers (one per utterance)
	:rtype: (`list`, `list`)
	"""
	preprocess_f = lru_cache(4)(preprocess_f) if preprocess_f else lambda x: x
	result_X = []
	result_y = []
	for e1, e2 in zip(episode.events, episode.events[1:]):
		if not _is_utterance(e1) or not _is_utterance(e2):
			raise NotImplementedError("Non-utterance Events are not supported yet")

		if e1.agent != e2.agent and e1.payload and e2.payload:
			result_X.append(preprocess_f(e1.payload))
			result_y.append(preprocess_f(e2.payload))

	return result_X, result_y

# Quick fix for circular dependencies
from due.event import Event
