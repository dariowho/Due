"""
Due is a learning, modular, action-oriented dialogue agent. `Agents` are the
entities that can take part in Episodes (:mod:`due.episode`), receiving and
issuing Events (:mod:`due.event`).
"""
import logging
import uuid
from abc import ABCMeta, abstractmethod
from datetime import datetime

import due
from due.event import Event
from due import episode as episode_module # TODO: check circular dependencies when just importing LiveEpisode 
from due.util.python import dynamic_import

class Agent(metaclass=ABCMeta):
	"""
	Participants in an Episodes are called Agents. An Agent models an unique
	identity through its ID, and can be served on a number of channels using
	packages in :mod:`due.serve`.

	Most importantly, Agent classes implement Natural Language Understanding
	(NLU) and Generation (NLG) models, which are the core of the whole
	conversational experience; they are meant to learn from Episodes coming from
	a corpus, as well as from live conversations with humans or other agents.

	:param agent_id: an unique ID for the Agent
	:type agent_id: `str`
	:param name: a human-friendly name for the Agent
	:type name: `str`
	"""

	def __init__(self, agent_id=None):
		self._logger = logging.getLogger(__name__)
		self.id = agent_id if agent_id is not None else str(uuid.uuid1())

	def __str__(self):
		return f"<Agent: {self.id}>"

	def save(self):
		"""
		Returns the Agent as an object. This object can be loaded with
		:func:`Agent.load` and can be (de)serialized using the
		:mod:`due.persistence` module.

		A saved Agent contains the following items:

		* `due_version`: version of the class who saved the agent (often `due.__version__`)
		* `class_name`: absolute import name of the Agent class (eg. `due.models.echo.EchoAgent`)
		* `data`: saved agent data. Will be passed to the Agent constructor's `_data` parameter

		NOTE: prior to version 0.1.dev6 serialization was handled differently.
		Legacy agents are not supported.

		:return: an object representing the Agent
		:rtype: object
		"""
		cls = type(self)
		return {
			'due_version': due.__version__,
			'class_name': f'{cls.__module__}.{cls.__name__}',
			'data': self.to_dict()
		}

	@staticmethod
	def load(saved_agent):
		"""
		Loads an Agent from an object that was produced with the :meth:`Agent.save`
		method.

		:param saved_agent: an Agent, as it was saved by :meth:`Agent.save`
		:type saved_agent: object
		:return: an Agent
		:rtype: `due.agent.Agent`
		"""
		class_ = dynamic_import(saved_agent['class_name'])
		assert issubclass(class_, Agent)
		return class_.from_dict(saved_agent['data'])

	def start_episode(self, other):
		"""
		Create a new :class:`due.episode.Episode` to engage another Agent in a
		new conversation.

		:param other_agent: The Agent you are inviting to the conversation.
		:type other_agent: :class:`due.agent.Agent`
		:return: a new Episode object
		:rtype: :class:`due.episode.LiveEpisode`
		"""
		result = episode_module.LiveEpisode(self, other)
		other.new_episode_callback(result)
		return result

	def new_episode_callback(self, new_episode):
		"""
		This is a callback method that is invoked whenever the Agent is invited
		to join a new conversation (Episode) with another one.

		Note that the base implementation is limited to logging the event:
		subclasses of :class:`Agent` should implement their own.

		:param new_episode: the new Episode that the other Agent has created
		:type new_episode: :class:`due.episode.Episode`
		"""
		self._logger.info("Agent %s invited to Episode %s", self, new_episode)

	#
	# Event Callbacks
	#

	def event_callback(self, episode, event):
		"""
		This is a callback method that is invoked whenever a new Event is acted
		in an Episode. This method acts as a proxy to specific Event type
		handlers:

		* :meth:`Agent.utterance_callback` (:class:`due.event.Event.Type.Utterance`)
		* :meth:`Agent.action_callback` (:class:`due.event.Event.Type.Action`)
		* :meth:`Agent.leave_callback` (:class:`due.event.Event.Type.Leave`)

		:param episode: The Episode where the Event was acted
		:type episode: :class:`due.episode.Episode`
		:param event: The new Event
		:type event: :class:`due.event.Event`
		:return: A list of response Events
		:rtype: `list` of :class:`due.event.Event`
		"""
		if event.type == Event.Type.Utterance:
			result = self.utterance_callback(episode, event)
		elif event.type == Event.Type.Action:
			result = self.action_callback(episode, event)
		elif event.type == Event.Type.Leave:
			result = self.leave_callback(episode, event)

		if not result:
			result = []

		return result

	def utterance_callback(self, episode, event):
		"""
		This is a callback method that is invoked whenever a new Utterance
		Event is acted in an Episode.

		:param episode: the Episode where the Utterance was acted
		:type episode: `due.episode.Episode`
		:param event: the Utterance Event that was acted in the Episode
		:type event: `due.event.Event`
		:return: A list of response Events
		:rtype: `list` of :class:`due.event.Event`
		"""
		self._logger.info("Agent %s received Utterance event %s in episode %s", self, event, episode)
		return []

	def action_callback(self, episode, event):
		"""
		This is a callback method that is invoked whenever a new Action Event
		is acted in an Episode.

		:param episode: the Episode where the Action was acted
		:type episode: `due.episode.Episode`
		:param event: the Action Event that was acted in the Episode
		:type event: `due.event.Event`
		:return: A list of response Events
		:rtype: `list` of :class:`due.event.Event`
		"""
		self._logger.info("Agent %s received Action event %s in episode %s", self, event, episode)
		return []

	def leave_callback(self, episode, event):
		"""
		This is a callback method that is invoked whenever a new Leave Event is
		acted in an Episode.

		:param episode: the Episode where the Leave Event was acted
		:type episode: `due.episode.Episode`
		:param event: the Leave Event that was acted in the Episode
		:type event: `due.event.Event`
		:return: A list of response Events
		:rtype: `list` of :class:`due.event.Event`
		"""
		self._logger.info("Agent %s received Leave event %s in episode %s", self, event, episode)
		return []

	#
	# Event Issuing
	#

	def act_events(self, events, episode):
		"""
		Act a sequence of Events in the given Episode.

		:param events: a list of Events
		:type events: `list` of :class:`due.event.Event`
		:param episode: an Episode
		:type episode: :class:`due.episode.Episode`
		"""
		for e in events:
			if e.type == Event.Type.Action:
				e.payload.run()

			episode.add_event(e)

	def say(self, sentence, episode):
		"""
		Create an Event out of the given sentence and act the new Event in
		the given Episode. :class:`Agent` subclassed may need to extend this
		implementation with some output operation (eg. print on screen,
		broadcast to a jabber chat...).

		:param sentence: A sentence
		:type sentence: :class:`str`
		:param episode: An Episode
		:type episode: :class:`due.episode.Episode`
		"""
		utterance_event = Event(Event.Type.Utterance, datetime.now(), self.id, sentence)
		episode.add_event(utterance_event)

	def do(self, action, episode):
		"""
		Create an Event out of the given Action and acts the new Event in the
		given Episode.

		:param action: An Action
		:type action: :class:`due.action.Action`
		"""
		action.run()
		action_event = Event(Event.Type.Action, datetime.now(), self.id, action)
		episode.add_event(action_event)

	def leave(self, episode):
		"""
		Acts a new Leave Event in the given Episode.

		:param episode: One of the Agent's active episodes
		:type episode: :class:`due.episode.Episode`
		"""
		leave_event = Event(Event.Type.Leave, datetime.now(), self.id, None)
		episode.add_event(leave_event)

	#
	# Abstract methods
	#

	@abstractmethod
	def learn_episodes(self, episodes):
		"""
		Submit a list of Episodes for the :class:`Agent` to learn.

		:param episodes: a list of episodes
		:type episodes: `list` of :class:`due.episode.Episode`
		"""
		pass

	@abstractmethod
	def to_dict(self):
		"""
		Create a `dict` representation of the :class:`Agent`, that can be
		interpreted by :meth:`from_dict` to build equivalent objects.

		NOTE: this method is functional to :meth:`save`for producing a saved
		copy of the agent.
		"""

	@staticmethod
	@abstractmethod
	def from_dict(data):
		"""
		Build an :class:`Agent` object out of its `dict` representation, as it
		was produced by :meth:`to_dict`.

		NOTE: this method is functional to :meth:`load` for restoring a saved
		copy of the agent.
		"""

class DummyAgent(Agent):

	def __init__(self):
		super().__init__()

	def learn_episodes(self, episodes):
		"""See :meth:`due.agent.Agent.from_dict`"""
		self._logger.warning("Dummy Agents don't learn...")

	def to_dict(self):
		"""See :meth:`due.agent.Agent.from_dict`"""
		return {'id': self.id}

	@staticmethod
	def from_dict(data):
		result = DummyAgent()
		result.id = data['id']
		return result
