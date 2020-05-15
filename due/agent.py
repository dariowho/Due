"""
Due is a learning, modular, action-oriented dialogue agent. `Agents` are the
entities that can take part in Episodes (:mod:`due.episode`), receiving and
issuing Events (:mod:`due.event`).
"""
import uuid
from abc import ABCMeta, abstractmethod
from datetime import datetime

from due.event import Event
from due import episode
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
		self.id = agent_id if agent_id is not None else str(uuid.uuid1())

	@abstractmethod
	def save(self):
		"""
		Returns the Agent as an object. This object can be loaded with
		:func:`Agent.load` and can be (de)serialized using the
		:mod:`due.persistence` module.

		A saved Agent must be a dictionary containing exactly the following items:

		* `version`: version of the class who saved the agent (often `due.__version__`)
		* `class`: absolute import name of the Agent class (eg. `due.models.dummy.DummyAgent`)
		* `data`: saved agent data. Will be passed to the Agent constructor's `_data` parameter

		:return: an object representing the Agent
		:rtype: object
		"""
		pass

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
		class_ = dynamic_import(saved_agent['class'])
		return class_(_data=saved_agent['data'])

	@abstractmethod
	def learn_episodes(self, episodes):
		"""
		Submit a list of Episodes for the :class:`Agent` to learn.

		:param episodes: a list of episodes
		:type episodes: `list` of :class:`due.episode.Episode`
		"""
		pass

	def learn_episode(self, episode):
		"""
		Submit an Episode for the Agent to learn. By default, this just wraps a
		call to :meth:`Agent.learn_episode`

		:param episode: an Episode
		:type episode: :class:`due.episode.Episode`
		"""
		self.learn_episodes([episode])

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

	def start_episode(self, other):
		"""
		Create a new :class:`due.episode.Episode` to engage another Agent in a
		new conversation.

		:param other_agent: The Agent you are inviting to the conversation.
		:type other_agent: :class:`due.agent.Agent`
		:return: a new Episode object
		:rtype: :class:`due.episode.LiveEpisode`
		"""
		result = episode.LiveEpisode(self, other)
		other.new_episode_callback(result)
		return result

	def event_callback(self, event, episode):
		"""
		This is a callback method that is invoked whenever a new Event is acted
		in an Episode. This method acts as a proxy to specific Event type
		handlers:

		* :meth:`Agent.utterance_callback` (:class:`due.event.Event.Type.Utterance`)
		* :meth:`Agent.action_callback` (:class:`due.event.Event.Type.Action`)
		* :meth:`Agent.leave_callback` (:class:`due.event.Event.Type.Leave`)

		:param event: The new Event
		:type event: :class:`due.event.Event`
		:param episode: The Episode where the Event was acted
		:type episode: :class:`due.episode.Episode`
		:return: A list of response Events
		:rtype: `list` of :class:`due.event.Event`
		"""
		if event.type == Event.Type.Utterance:
			result = self.utterance_callback(episode)
		elif event.type == Event.Type.Action:
			result = self.action_callback(episode)
		elif event.type == Event.Type.Leave:
			result = self.leave_callback(episode)

		if not result:
			result = []
		
		return result

	@abstractmethod
	def utterance_callback(self, episode):
		"""
		This is a callback method that is invoked whenever a new Utterance
		Event is acted in an Episode.

		:param episode: the Episode where the Utterance was acted
		:type episode: `due.episode.Episode`
		:return: A list of response Events
		:rtype: `list` of :class:`due.event.Event`
		"""
		pass

	@abstractmethod
	def action_callback(self, episode):
		"""
		This is a callback method that is invoked whenever a new Action Event
		is acted in an Episode.

		:param episode: the Episode where the Action was acted
		:type episode: `due.episode.Episode`
		:return: A list of response Events
		:rtype: `list` of :class:`due.event.Event`
		"""
		pass

	@abstractmethod
	def leave_callback(self, episode):
		"""
		This is a callback method that is invoked whenever a new Leave Event is
		acted in an Episode.

		:param episode: the Episode where the Leave Event was acted
		:type episode: `due.episode.Episode`
		:return: A list of response Events
		:rtype: `list` of :class:`due.event.Event`
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

	def __str__(self):
		return f"<Agent: {self.id}>"
