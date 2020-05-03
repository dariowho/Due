"""
Serve a Due :class:`due.agent.Agent` through an interactive Command Line Interface (CLI)
"""
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit import print_formatted_text, HTML

from due.models.dummy import DummyAgent
from due.episode import LiveEpisode, AsyncLiveEpisode

style = Style.from_dict({
    'human':     '#00ffff bg:#444400',
    'agent':     '#ff1111 bg:#444400',
})

class ConsoleLiveEpisode(AsyncLiveEpisode):

    def __init__(self, live_episode, agents_noecho=None):
        super().__init__(live_episode.starter, live_episode.invited)
        self.agents_noecho = [a.id for a in agents_noecho] if agents_noecho else []

    def add_event(self, event):
        if event.agent not in self.agents_noecho:
            print_formatted_text(HTML(f'<agent>{event.agent} ></agent> {event.payload}'), style=style)

        super().add_event(event)

def serve(agent):
    human = DummyAgent('human')

    live_episode = human.start_episode(agent)
    live_episode = ConsoleLiveEpisode(live_episode, [human])
    text = ''
    while text != '!q':
        text = prompt(HTML(f'<human>{human.id} ></human> '), style=style)
        human.say(text, live_episode)
        # print_formatted_text(HTML(f'<agent>{agent.id} ></agent> You said <i>"{text}"</i>'), style=style)
