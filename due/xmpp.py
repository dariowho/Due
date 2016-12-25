from due.agent import Due, HumanAgent
from due.episode import Episode

import logging
import uuid

from sleekxmpp import ClientXMPP
from sleekxmpp.exceptions import IqError, IqTimeout


class DueBot(Due, ClientXMPP):

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
		Due.__init__(self)
		ClientXMPP.__init__(self, jid, password)

		self._humans = {}
		self._current_episode = None
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
			if self._current_episode is None:
				self._current_episode = Episode(human_agent, self)
			self._current_episode.add_utterance(human_agent, msg['body'])

	def say(self, sentence, episode):
		if self._last_message is None:
			self._logger.warning("Could not send message '"+sentence+"' because \
				no last message is set. This is not supposed to happen.")
			return
		self._last_message.reply(sentence).send()
		episode.add_utterance(self, sentence)

	def _fetch_or_create_human_agent(self, jid):
		if jid not in self._humans:
			self._humans[jid] = HumanAgent(uuid.uuid1, jid)
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
			self._current_episode.leave(human_agent)
			self._current_episode = None
			self._last_message = None
		return True
