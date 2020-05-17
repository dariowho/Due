"""
Serve a Due :class:`due.agent.Agent` as a Telegram bot.

Note that this module is not stateless, and as of now it cannot be scaled horizontally.
"""
import logging
from functools import partial

import telegram.ext as tg

from due import Event
from due.agent import DummyAgent
from due.episode import LiveEpisode

# from due.util.python import dynamic_import
# Pylint complains because lib has the same name of this module
# tg = dynamic_import('telegram.ext')

logger = logging.getLogger(__name__)

class TelegramLiveEpisode(LiveEpisode):
	"""
	This is a helper class that sends incoming Events to a Telegram Channel.

	:param live_episode: An existing :class:`due.episode.LiveEpisode` to start from
	:type live_episode: :class:`due.episode.LiveEpisode`
	:param agents_noecho: Events coming from these agents will not be printed on screen, defaults to None
	:type agents_noecho: `list` of :class:`due.agent.Agent`, optional
	"""

	def __init__(self, live_episode, human):
		super().__init__(live_episode.starter, live_episode.invited)
		self.human = human
		self.last_update = None

	def echo_event(self, event):
		logger.info("Echo Event: %s", event)

		if event.type != Event.Type.Utterance:
			logger.warning("Skipping non-utterance event")
			return

		if not self.last_update:
			logger.warning("No last update set, skipping event: %s", event)
			return

		if event.agent != self.human.id:
			logger.info("replying")
			self.last_update.message.reply_text(event.payload)

def help_command(update, context):
	update.message.reply_text('Welcome! Chat with this bot as you would with any other contact :) There are no special commands yet.')

def handle_message(update, context, live_episodes):
	"""Echo the user message."""
	logging.info("update: %s", update)
	episode = live_episodes.get(update.effective_user.id)
	episode.last_update = update
	episode.human.say(update.message.text, episode)

def error(update, context):
	"""Log Errors caused by Updates."""
	logger.error('Update "%s" caused error "%s"', update, context.error)

class LiveEpisodeCache():

	def __init__(self, agent):
		self.agent = agent
		self.cache = {}

	def get(self, user_id):
		if user_id not in self.cache:
			human = DummyAgent()
			live_episode = human.start_episode(self.agent)
			self.cache[user_id] = TelegramLiveEpisode(live_episode, human)

		return self.cache[user_id]

def serve(agent, telegram_token):
	"""
	Serve an agent on Telegram.

	Telegram updates have the following structure:

	.. code-block:: json

		{
			"update_id": 551804904,
			"message": {
				"message_id": 7,
				"date": 1589613843,
				"chat": {
					"id": 12345678,
					"type": "private",
					"first_name": "Dario"
				},
				"text": "Hello",
				"entities": [],
				"caption_entities": [],
				"photo": [],
				"new_chat_members": [],
				"new_chat_photo": [],
				"delete_chat_photo": false,
				"group_chat_created": false,
				"supergroup_chat_created": false,
				"channel_chat_created": false,
				"from": {
					"id": 12345678,
					"first_name": "Dario",
					"is_bot": false,
					"language_code": "en"
				}
			},
			"_effective_user": {
				"id": 12345678,
				"first_name": "Dario",
				"is_bot": false,
				"language_code": "en"
			},
			"_effective_chat": {
				"id": 12345678,
				"type": "private",
				"first_name": "Dario"
			},
			"_effective_message": {
				"message_id": 7,
				"date": 1589613843,
				"chat": {
					"id": 12345678,
					"type": "private",
					"first_name": "Dario"
				},
				"text": "Hello",
				"entities": [],
				"caption_entities": [],
				"photo": [],
				"new_chat_members": [],
				"new_chat_photo": [],
				"delete_chat_photo": false,
				"group_chat_created": false,
				"supergroup_chat_created": false,
				"channel_chat_created": false,
				"from": {
					"id": 12345678,
					"first_name": "Dario",
					"is_bot": false,
					"language_code": "en"
				}
			}
		}

	:param agent: The Agent to serve
	:type agent: :class:`due.Agent`
	:param telegram_token: A token for a Telegram bot
	:type telegram_token: `str`
	"""
	updater = tg.Updater(telegram_token, use_context=True)

	dp = updater.dispatcher

	# on different commands - answer in Telegram
	dp.add_handler(tg.CommandHandler("help", help_command))

	# on noncommand i.e message - echo the message on Telegram
	live_episodes = LiveEpisodeCache(agent)
	dp.add_handler(tg.MessageHandler(tg.Filters.text, partial(handle_message, live_episodes=live_episodes)))

	# log all errors
	dp.add_error_handler(error)

	# Start the Bot
	updater.start_polling()

	# Run the bot until you press Ctrl-C or the process receives SIGINT,
	# SIGTERM or SIGABRT. This should be used most of the time, since
	# start_polling() is non-blocking and will stop the bot gracefully.
	updater.idle()
