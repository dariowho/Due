"""
Serve a Due :class:`due.agent.Agent` through an interactive Command Line Interface (CLI)
"""
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit import print_formatted_text, HTML

from due.models.dummy import DummyAgent
from due.episode import AsyncLiveEpisode
from due.util.capture_io import CaptureIO

style = Style.from_dict({
	'human':     '#00ffff bg:#444400',
	'agent':     '#ff1111 bg:#444400',
})

class ConsoleLiveEpisode(AsyncLiveEpisode):
	"""
	This is a helper class that prints incoming events on screen.

	:param live_episode: An existing :class:`due.episode.LiveEpisode` to start from
	:type live_episode: :class:`due.episode.LiveEpisode`
	:param agents_noecho: Events coming from these agents will not be printed on screen, defaults to None
	:type agents_noecho: `list` of :class:`due.agent.Agent`, optional
	"""

	def __init__(self, live_episode, agents_noecho=None):
		super().__init__(live_episode.starter, live_episode.invited)
		self.agents_noecho = [a.id for a in agents_noecho] if agents_noecho else []

	def add_event(self, event):
		if event.agent not in self.agents_noecho:
			print_formatted_text(HTML(f'<agent>{event.agent} ></agent> {event.payload}'), style=style)

		super().add_event(event)

def serve(agent):
	"""
	Serve an agent on a very basic terminal-based chat interface.

	**NOTE** that Event handling is asynchronous. To prevent interferences with
	user input, incoming logs are buffered and released at each conversation turn.

	:param agent: The Agent to serve
	:type agent: :class:`due.Agent`
	"""
	human = DummyAgent('human')

	live_episode = human.start_episode(agent)
	live_episode = ConsoleLiveEpisode(live_episode, [human])
	text = ''

	with CaptureIO(logging_only=True) as captured_logs:
		while text != '!q':
			for line in captured_logs.flush():
				print(line)
			text = prompt(HTML(f'<human>{human.id} ></human> '), style=style)
			human.say(text, live_episode)

	for line in captured_logs.flush():
		print(line)
