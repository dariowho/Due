import logging
import uuid
from datetime import datetime

from due.models.tfidf import TfIdfAgent
from due.models.dummy import DummyAgent
from due.episode import LiveEpisode
from due.event import Event

from sleekxmpp import ClientXMPP
from sleekxmpp.exceptions import IqError, IqTimeout


class DueBot(TfIdfAgent, ClientXMPP):

	DEFAULT_HUMAN_JID = "default@human.im"

	def __init__(self, jid, password):
		"""
		Creates a DueBot object with the given XMPP credentials.

		Example:

			>>> past_episodes = ... # a list of past episodes
			>>> bot = DueBot("due@xmppprovider.net", "p4ssw0rd")
			>>> bot.learn_episodes(past_episodes)
			>>> bot.connect()
			>>> bot.process(block=True)

		:param jid: A Jabber ID
		:type jid: :class:`str`
		:param password: The Jabber account password
		:type password: :class:`str`
		"""
		TfIdfAgent.__init__(self)
		ClientXMPP.__init__(self, jid, password)

		self._humans = {}
		self._live_episode = None
		self._last_message = None
		self._logger = logging.getLogger(__name__ + ".DueBot")

		self.add_event_handler("session_start", self.session_start)
		self.add_event_handler("message", self.message)

	def session_start(self, event):
		self.send_presence()
		self.get_roster()

	def message(self, msg):
		if msg['type'] in ('chat', 'normal'):
			human_agent = self._fetch_or_create_human_agent(DueBot.DEFAULT_HUMAN_JID)
			self._last_message = msg
			if self._handle_command_message(msg):
				return
			if self._live_episode is None:
				self._live_episode = LiveEpisode(human_agent, self)
			utterance = Event(Event.Type.Utterance, datetime.now(), str(human_agent.id), msg['body'])
			self._live_episode.add_event(human_agent, utterance)

	def utterance_callback(self, episode):
		"""See :meth:`due.agent.Agent.utterance_callback`"""
		self._logger.debug("Received utterance")
		answers =super().utterance_callback(episode)
		self.act_events(answers, episode)

	def act_events(self, events, episode):
		for e in events:
			if e.type == Event.Type.Action:
				e.payload.run()
			elif e.type == Event.Type.Utterance:
				if self._last_message is None:
					self._logger.warning("Could not send message '%s' because \
						no last message is set. This is not supposed to happen.", e.payload)
					return
				self._last_message.reply(e.payload).send()

			episode.add_event(self, e)

	def say(self, sentence, episode):
		"""
		Deprecated!
		"""
		self._logger.warn('deprecated use of DueBot.say(). Please use DueBot.act_events() instead.')
		if self._last_message is None:
			self._logger.warning("Could not send message '%s' because \
				no last message is set. This is not supposed to happen.", sentence)
			return
		self._last_message.reply(sentence).send()
		episode.add_utterance(self, sentence)

	def _fetch_or_create_human_agent(self, jid):
		if jid not in self._humans:
			self._humans[jid] = DummyAgent(jid)
		return self._humans[jid]

	def _handle_command_message(self, msg):
		"""
		Command messages are:

			* `,,,leave`: closes the episode
		"""
		if msg['body'][0:3] != ',,,': return False
		if msg['body'] == ',,,leave':
			msg.reply("[you left the episode]").send()
			human_agent = self._fetch_or_create_human_agent(DueBot.DEFAULT_HUMAN_JID)
			leave_event = Event(Event.Type.Leave, datetime.now(), str(human_agent.id), None)
			self._live_episode.add_event(human_agent, leave_event)
			self._live_episode = None
			self._last_message = None
		return True
