"""
This module loads the Cornell Movie-Dialog Corpus. More detail at
http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

There are some givens to consider when using this corpus:

* No **timestamp** information is provided in the dataset. The first returned \
episode symbolically starts the day the paper was published, 15/6/2011, at noon, \
all the following events through the episodes are placed 1 second apart one \
another.
* The corpus only contains **one on one** conversations only.
* Context-wise, episodes are **not self-contained**. That is, an episode in the \
corpus may start in the middle of a longer conversation, and may end before \
that conversation is finished.
"""

import re
import ast
from datetime import datetime, timedelta

import pandas as pd

from due.util.python import is_notebook
if is_notebook():
	from tqdm import tqdm_notebook as tqdm
else:
	import tqdm

from due.episode import Episode
from due.event import Event
from due import resource_manager

rm = resource_manager

START_DATE = datetime(2011, 6, 15, 12, 0)

def load():
	return list(episodes())

def episodes():
	"""
	Returns the Cornell Movie-Dialog Corpus as a list of
	:class:`due.episode.Episode`\ s

	TODO: return agents dictionary

	:return: The Cornell Movie-Dialog Corpus
	:rtype: `list` of :class:`due.episode.Episode`
	"""
	global current_date

	with _open_cornell('movie_conversations.txt') as f:
		df_conversations = _read_cornell(f, ['character_id_A', 'character_id_B', 'movie_id', 'utterance_list'])
		df_conversations['utterance_list'] = df_conversations['utterance_list'].apply(ast.literal_eval)

	# with _open_cornell('movie_characters_metadata.txt') as f:
	# 	df_characters = _read_cornell(f, ['character_id', 'character_name', 'movie_id', 'movie_title', 'gender', 'credits_position'])

	with _open_cornell('movie_lines.txt') as f:
		df_lines = _read_cornell(f, ['line_id', 'character_id', 'movie_id', 'character_name', 'utterance'])
		df_lines.set_index('line_id', inplace=True)

	current_date = START_DATE
	for conversation in tqdm(df_conversations.itertuples(), total=len(df_conversations)):
		events = [_build_event(l_id, df_lines) for l_id in conversation.utterance_list]
		episode = _build_episode(conversation, events)
		yield episode


#
# Helpers
#

def _open_cornell(filename):
	filename_full = 'cornell movie-dialogs corpus/%s' % filename
	return rm.open_resource_file('corpora.cornell', filename_full, binary=False, encoding='iso-8859-1')

def _read_cornell(file_buffer, columns):
	return pd.read_csv(file_buffer, sep=re.escape(' +++$+++ '), names=columns, engine='python')

current_date = START_DATE
def _build_event(line_id, df_lines):
	global current_date
	line = df_lines.loc[line_id]
	result = Event(Event.Type.Utterance, current_date, line.character_id, line.utterance)
	current_date += timedelta(seconds=1)
	return result

def _build_episode(conversation, events):
	agents = set([conversation.character_id_A, conversation.character_id_B])
	starter_agent = events[0].agent
	invited_agent = (agents - set([starter_agent])).pop()
	result = Episode(starter_agent, invited_agent)
	result.events = events
	return result
