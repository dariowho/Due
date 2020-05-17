"""
An Episode is a sequence of Events issued by agents. Here we define an interface
for Episodes, as well as some helper methods to manipulate their content:

	* :class:`Episode` models recorded Episodes that can be used to train agents
	* :class:`LiveEpisode` models Episodes that are still in progress.
	* :func:`extract_utterance_pairs` will extract utterances as strings from Episodes.

API
===
"""
import io
import json
import uuid
import asyncio
import logging
from functools import lru_cache
from datetime import datetime

import numpy as np
import pandas as pd

import due.agent
from due.util.time import convert_datetime, parse_timedelta

UTTERANCE_LABEL = 'utterance'
MAX_EVENT_RESPONSES = 200

class Episode(object):
	"""
	An Episode is a sequence of Events issued by Agents
	"""

	def __init__(self, starter_agent_id, invited_agent_id):
		self._logger = logging.getLogger(__name__ + ".Episode")
		self.starter_id = starter_agent_id
		self.invited_id = invited_agent_id
		self.id = str(uuid.uuid1())
		self.timestamp = datetime.now()
		self.events = []

	def __eq__(self, other):
		if isinstance(other, Episode):
			if self.starter_id != other.starter_id: return False
			if self.invited_id != other.invited_id: return False
			if self.id != other.id: return False
			if self.timestamp != other.timestamp: return False
			if self.events != other.events: return False
			return True

		return False

	def __ne__(self, other):
		return not self.__eq__(other)

	def last_event(self, event_type=None):
		"""
		Returns the last event in the Episode. Optionally, events can be filtered
		by type.

		:param event_type: an event type, or a collection of types
		:type event_type: :class:`Event.Type` or list of :class:`Event.Type`
		"""
		event_type = event_type if not isinstance(event_type, Event.Type) else (event_type,)
		for e in reversed(self.events):
			if event_type is None or e.type in event_type:
				return e
		return None

	def save(self, output_format='standard'):
		"""
		Save the Episode to a serializable object, that can be loaded with
		:meth:`due.episode.Episode.load`.

		By default, episodes are saved in the **standard** format, which is a
		dict of metadata with a list of saved Events, which format is handled by
		the :class:`due.event.Event` class).

		It is also possible to save the Episode in the **compact** format. In
		compact representation, event objects are squashed into CSV lines. This
		makes them slower to load and save, but more readable and easily
		editable without the use of external tools; because of this, they are
		especially suited for toy examples and small hand-crafted corpora.

		:return: a serializable representation of `self` :rtype: `dict`
		"""
		result = {
			'id': self.id,
			'timestamp': self.timestamp,
			'starter_agent': str(self.starter_id),
			'invited_agents': [str(self.invited_id)],
			'events': [e.save() for e in self.events],
			'format': 'standard'
		}

		if output_format == 'compact':
			return _compact_saved_episode(result)

		return result

	@staticmethod
	def load(saved_episode):
		"""
		Loads an Episode as it was saved by :meth:`due.episode.Episode.save`.

		:param saved_episode: the episode to be loaded
		:type saved_episode: `dict`
		:return: an Episode object representing `saved_episode`
		:rtype: :class:`due.episode.Episode`
		"""
		if saved_episode['format'] == 'compact':
			saved_episode = _uncompact_saved_episode(saved_episode)

		result = Episode(saved_episode['starter_agent'], saved_episode['invited_agents'][0])
		result.id = saved_episode['id']
		result.timestamp = convert_datetime(saved_episode['timestamp'])
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
		self._logger = logging.getLogger(__name__ + ".LiveEpisode")
		self.starter = starter_agent
		self.invited = invited_agent
		self._agent_by_id = {
			starter_agent.id: starter_agent,
			invited_agent.id: invited_agent
		}

	def add_event(self, event):
		"""
		Adds an Event to the LiveEpisode, triggering the
		:meth:`due.agent.Agent.event_callback` method on the other participants.
		Response Events that are returned from the callback which will be
		processed iteratively.

		:param agent: the agent which acted the Event
		:type agent: :class:`due.agent.Agent`
		:param event: the event that was acted by the Agent
		:type event: :class:`due.event.Event`
		"""
		new_events = [event]

		count = 0
		while new_events:
			e = new_events.pop(0)
			self.echo_event(e)
			agent = self.agent_by_id(e.agent)
			self.events.append(e)
			e.mark_acted()
			for a in self._other_agents(agent):
				self._logger.info("Notifying %s", a)
				response_events = a.event_callback(e, self)
				new_events.extend(response_events)

			count += 1
			if count > MAX_EVENT_RESPONSES:
				self._logger.warning("Agents reached maximum number of responses allowed for a single Event (%s). Further Events won't be notified to Agents", MAX_EVENT_RESPONSES)
				break

		self.events.extend(new_events)
		[e.mark_acted() for e in new_events]

	def echo_event(self, event):
		"""
		Echoes an event on an output stream. This just logs to screen by
		default.
		"""
		self._logger.info("New event: '%s'", event)

	def agent_by_id(self, agent_id):
		"""
		Retrieve the :class:`due.agent.Agent` object of one of the agents that
		are participating in the :class:`LiveEpisode`. Raise `ValueError` if the
		given ID does not correspond to any of the agents in the Episode.

		:param agent_id: ID of one of the agents in the LiveEpisode
		:type agent_id: :class:`due.agent.Agent`
		"""
		if agent_id not in self._agent_by_id:
			raise ValueError(f"Agent '{agent_id}' not found in LiveEpisode {self}")

		result = self._agent_by_id[agent_id]
		assert isinstance(result, due.agent.Agent)
		return result

	def _other_agents(self, agent):
		return [self.starter] if agent == self.invited else [self.invited]

class AsyncLiveEpisode(LiveEpisode):
	"""
	This is a subclass of :class:`LiveEpisode` that implement asynchronous
	notification of new Events.
	"""

	def add_event(self, event):
		self.events.append(event)
		event.mark_acted()
		agent = self.agent_by_id(event.agent)
		for a in self._other_agents(agent):
			loop = asyncio.get_event_loop()
			loop.create_task(self.async_event_callback(a, event))

	async def async_event_callback(self, agent, event):
		self._logger.info("Notifying event %s to agent %s", event, agent)
		response_events = agent.event_callback(event, self)
		if response_events:
			for e in response_events:
				self.add_event(e)

#
# Save/Load Helpers
#

def _compact_saved_episode(saved_episode):
	"""
	Convert a saved episode into a compact representation.
	"""
	events = saved_episode['events']
	events = [_compact_saved_event(e) for e in events]
	df = pd.DataFrame(events)
	s = io.StringIO()
	df.to_csv(s, sep='|', header=False, index=False)
	compact_events = [l for l in s.getvalue().split('\n') if l]
	return {**saved_episode, 'events': compact_events, 'format': 'compact'}

def _compact_saved_event(saved_event):
	"""
	Prepare an Event for compact serialization, meaning that its fields must
	be writable as a line of CSV). This is always the case, except for Actions,
	which payloads are objects. In this case, we serialize them as JSON.
	"""
	e = saved_event
	if e['type'] == Event.Type.Action.value:
		return {**e, 'payload': json.dumps(e['payload'])}
	return e

def _uncompact_saved_episode(compact_episode):
	"""
	Convert a compacted saved episode back to the standard format.
	"""
	buf = io.StringIO('\n'.join(compact_episode['events']))
	df = pd.read_csv(buf, sep='|', names=['type', 'timestamp', 'agent', 'payload'])
	compact_events = df.replace({np.nan:None}).to_dict(orient='records')
	events = []
	last_timestamp = convert_datetime(compact_episode['timestamp'])
	for e in compact_events:
		e_new = _uncompact_saved_event(e, last_timestamp)
		events.append(e_new)
		last_timestamp = e_new['timestamp']
	return {**compact_episode, 'events': events, 'format': 'standard'}

def _uncompact_saved_event(compact_event, last_timestamp):
	"""
	Note that `compact_event` is not the CSV line. It is already its dict
	representation, but Action payloads need to be deserialized from JSON.
	Also, the Pandas interpretation of CSV needs to be fixed, by converting
	timestamps to `datetime`, and converting NaN values to `None`.
	"""
	e = compact_event
	timestamp = _uncompact_timestamp(compact_event, last_timestamp)
	e = {**e, 'timestamp': timestamp}
	if compact_event['type'] == Event.Type.Action.value:
		e['payload'] = json.loads(e['payload'])
	return e


def _uncompact_timestamp(compact_event, last_timestamp):
	"""
	In compacted episodes the timestamp can be a ISO string, or as a time
	difference from the previous event; in this latter case, the delta must be
	expressed as a int (number of seconds) or in the `1d2h3m4s` format (see
	:func:`due.util.time.parse_timedelta`).
	"""
	try:
		return convert_datetime(compact_event['timestamp'])
	except ValueError:
		return last_timestamp + parse_timedelta(compact_event['timestamp'])

#
# Utilities
#

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
