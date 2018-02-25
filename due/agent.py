"""
Due is a learning, modular, action-oriented dialogue agent. `Agents` are the 
entities that can take part in Episodes (:mod:`due.episode`), receiving and 
issuing Events (:mod:`due.event`) with the optional use of a Natural Language
Understanding and Generation model (Brain, :mod:`due.brain`).

This module defines the base class for Agents (:class:`due.agent.Agent`), as
well as two implementations:

* :class:`due.agent.HumanAgent`: a dummy agent that just logs down the Episodes
  and Events it receives
* :class:`due.agent.Due`: an autonomous agent that uses a :class:`due.brain.Brain`
  to understand Events and generate new ones.
"""
import uuid
from abc import ABCMeta, abstractmethod
from datetime import datetime
import logging

from due.episode import LiveEpisode
from due.event import Event
from due.brain import Brain
from due.brain import CosineBrain

class Agent(metaclass=ABCMeta):
	"""
	Participants in an Episodes are called Agents. An Agent models an unique
	identity through its ID, and handles the communication on the channel level.

	Optionally, a human-friendly name can be provided, which should not be taken
	into account in Machine Learning processes.

	:param id: an unique ID for the Agent
	:type id: object
	:param name: a human-friendly name for the Agent
	:type name: `str`
	"""

	def __init__(self, id=None, name=None):
		self.id = id if id is not None else uuid.uuid1()
		self.name = name

	@abstractmethod
	def save(self):
		"""
		Returns the Agent as an object. This object can be loaded with 
		:func:`Agent.load` and can be (de)serialized using the 
		:mod:`due.persistence` module.

		Note that this is an **abstract method**: subclasses of :class:`Agent` 
		must implement their own.

		:return: an object representing the Agent
		:rtype: object
		"""
		pass

	@staticmethod
	@abstractmethod
	def load(saved_agent):
		"""
		Loads an Agent from an object produced with the :meth:`Agent.save` 
		method.

		Note that this is an **abstract method**: subclasses of :class:`Agent` 
		must implement their own.

		:param saved_agent: an Agent, as it was saved by :meth:`Agent.save`
		:type saved_agent: object
		:return: an Agent
		:rtype: `due.agent.Agent`
		"""
		pass

	@abstractmethod
	def new_episode_callback(self, new_episode):
		"""
		This is a callback method that is invoked whenever the Agent is invited
		to join a new conversation (Episode) with another one.

		Note that this is an **abstract method**: subclasses of :class:`Agent` 
		must implement their own. 

		:param new_episode: the new Episode that the other Agent has created
		:type new_episode: :class:`due.episode.Episode`
		"""
		pass

	def event_callback(self, event, episode):
		"""
		This is a callback method that is invoked whenever a new Event is acted
		in an Episode. This method acts as a proxy to specific Event type
		handlers:

		* :meth:`Agent.utterance_callback` (:class:`due.event.Event.Type.Utterance`)
		* :meth:`Agent.action_callback` (:class:`due.event.Event.Type.Action`)
		* :meth:`Agent.leave_callback` (:class:`due.event.Event.Type.Leave`)

		:param event: the new Event
		:type event: :class:`due.event.Event`
		:param episode: the Episode where the Event was acted
		:type episode: :class:`due.episode.Episode`
		"""
		if event.type == Event.Type.Utterance:
			self.utterance_callback(episode)
		elif event.type == Event.Type.Action:
			self.action_callback(episode)
		elif event.type == Event.Type.Leave:
			self.leave_callback(episode)


	@abstractmethod
	def utterance_callback(self, episode):
		"""
		This is a callback method that is invoked whenever a new Utterance
		Event is acted in an Episode.

		Note that this is an **abstract method**: subclasses of :class:`Agent` 
		must implement their own. 

		:param episode: the Episode where the Utterance was acted
		:type episode: `due.episode.Episode`
		"""
		pass

	@abstractmethod
	def action_callback(self, episode):
		"""
		This is a callback method that is invoked whenever a new Action Event
		is acted in an Episode.

		Note that this is an **abstract method**: subclasses of :class:`Agent` 
		must implement their own. 

		:param episode: the Episode where the Action was acted
		:type episode: `due.episode.Episode`
		"""
		pass

	@abstractmethod
	def leave_callback(self, episode):
		"""
		This is a callback method that is invoked whenever a new Leave Event is
		acted in an Episode.

		Note that this is an **abstract method**: subclasses of :class:`Agent` 
		must implement their own. 

		:param episode: the Episode where the Leave Event was acted
		:type episode: `due.episode.Episode`
		"""
		pass

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

			episode.add_event(self, e)

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
		episode.add_event(self, utterance_event)

	def do(self, action, episode):
		"""
		Create an Event out of the given Action and acts the new Event in the
		given Episode.

		:param action: An Action
		:type action: :class:`due.action.Action`
		"""
		action.run()
		action_event = Event(Event.Type.Action, datetime.now(), self.id, action)
		episode.add_event(self, action_event)

	def leave(self, episode):
		"""
		Acts a new Leave Event in the given Episode.

		:param episode: One of the Agent's active episodes
		:type episode: :class:`due.episode.Episode`
		"""
		leave_event = Event(Event.Type.Leave, datetime.now(), self.id, None)
		episode.add_event(self, leave_event)

	def __str__(self):
		name = self.name if self.name is not None else self.id
		return "<Agent: " + name + ">"

class HumanAgent(Agent):
	"""
	A Human Agent is an Agent that uses a Human brain, typically sitting behind
	a keyboard, to make sense of :class:`Episode`\ s and produce new
	:class:`.Event`\ s.

	This is the simplest kind of Agent, as it simply logs new Episodes and
	Events, expecting the interaction to be commanded externally.
	"""
	def __init__(self, id=None, name=None):
		super().__init__(id, name)
		self._active_episodes = {}
		self._logger = logging.getLogger(__name__ + '.HumanAgent')

	def save(self):
		"""See :meth:`due.agent.Agent.save`"""
		return {'id': self.id, 'name': self.name}

	@staticmethod
	def load(saved_agent):
		"""See :meth:`due.agent.Agent.load`"""
		return HumanAgent(**saved_agent)

	def start_episode(self, other):
		"""
		Engages another Agent in a new conversation, thus creating a new
		Episode.

		:param other_agent: The Agent you are inviting to the conversation.
		:type other_agent: :class:`due.agent.Agent`
		:return: a new Episode object
		:rtype: :class:`due.episode.LiveEpisode`
		"""
		result = LiveEpisode(self, other)
		other.new_episode_callback(result)
		return result

	def new_episode_callback(self, new_episode):
		"""See :meth:`due.agent.Agent.new_episode_callback`"""
		self._logger.info("New episode callback: " + str(new_episode))
		self._active_episodes[new_episode.id] = new_episode

	def utterance_callback(self, episode):
		"""See :meth:`due.agent.Agent.utterance_callback`"""
		self._logger.debug("Utterance received.")

	def action_callback(self, episode):
		"""See :meth:`due.agent.Agent.action_callback`"""
		self._logger.debug("Action received.")

	def leave_callback(self, episode):
		"""See :meth:`due.agent.Agent.leave_callback`"""
		agent = episode.last_event(Event.Type.Leave).agent
		self._logger.debug("Agent %s left the episode." % agent)

DEFAULT_DUE_UUID = uuid.UUID('423cc038-bfe0-11e6-84d6-a434d9562d81')

class Due(Agent):
	"""
	Due is a conversational agent that makes use of a :class:`due.brain.Brain`
	object to understand events and produce new ones.

	Due is a subclass of :class:`due.agent.Agent`.
	"""

	def __init__(self, id=DEFAULT_DUE_UUID, brain=None):

		Agent.__init__(self, id, "Due")
		
		if isinstance(brain, dict):
			brain = Brain.load(brain)

		brain = brain if brain is not None else CosineBrain()
		self._brain = brain
		self._logger = logging.getLogger(__name__ + ".Due")

	def learn_episodes(self, episodes):
		"""
		Trains the Natural Language and Generation (Brain) model with the
		given sequence of Episodes.

		:param episodes: a list of Episodes
		:type episodes: `list` of `due.episode.Episode`
		"""
		self._logger.info("Learning some episodes.")
		self._brain.learn_episodes(episodes)

	def new_episode_callback(self, episode):
		"""See :meth:`due.agent.Agent.new_episode_callback`"""
		self._logger.info("Got invited to a new episode.")
		self._brain.new_episode_callback(episode)

	def utterance_callback(self, episode):
		"""See :meth:`due.agent.Agent.utterance_callback`"""
		self._logger.debug("Received utterance")
		answers = self._brain.utterance_callback(episode)
		self.act_events(answers, episode)

	def action_callback(self, episode):
		"""See :meth:`due.agent.Agent.action_callback`"""
		self._logger.debug("Received action")

	def leave_callback(self, episode):
		"""See :meth:`due.agent.Agent.leave_callback`"""
		agent = episode.last_event(Event.Type.Leave).agent
		self._logger.debug("Agent %s left the episode." % agent)
		self._brain.leave_callback(episode, agent)

	@staticmethod
	def load(saved_agent):
		"""See :meth:`due.agent.Agent.load`"""
		return Due(saved_agent['id'], saved_agent['brain'])

	def save(self):
		"""See :meth:`due.agent.Agent.save`"""
		return {
			'id': self.id,
			'name': self.name,
			'brain': self._brain.save()
		}