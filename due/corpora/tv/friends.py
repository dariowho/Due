"""
This module can be used to load the Friends Corpus, by David Ayliffe. The
corpus contains conversations from the TV show "Friends":

https://sites.google.com/site/friendstvcorpus/

Conversations are grouped in **scenes** and each scene is given a brief English
description (which is currently not modelled in Episodes).
"""
import logging
from datetime import datetime, timedelta

import pandas as pd

from due.event import Event, EventTuple
from due.episode import Episode
from due import resource_manager

rm = resource_manager
logger = logging.getLogger(__name__)

START_DATE = datetime(year=1994, month=9, day=22)
DELTA = timedelta(seconds=2)

def episode_generator():
    """
    Return the list of all the Episodes in the Friends Corpus
    """
    with rm.open_resource_file('corpora.tv.friends', 'friends-final.txt', binary=False) as f:
        df = pd.read_csv(f, sep='\t', header=None, index_col=0)

    df.columns = ["scene_id", "person", "gender", "original_line", "line", "metadata", "filename"]
    scenes = [int(x) for x in set(df['scene_id']) if x.isdigit()]
    scenes.sort(key=lambda x: int(x))

    for i, scene_id in enumerate(scenes):
        try:
            scene_df = df[df['scene_id'] == str(scene_id)]
            episode = build_episode(scene_df)
            yield episode
        except Exception as e:
            logger.warning("Skipping scene %s because of exception: %s", i, e)

def _get_events_dataframe(df):
    df_events = df[['person', 'line']]
    df_events.columns = ['agent', 'payload']
    timestamp_series = pd.Series(pd.date_range(start=START_DATE, periods=len(df_events), freq=DELTA), index=df_events.index, name='timestamp')    
    df_events = df_events.join(pd.DataFrame(timestamp_series))
    df_events['type'] = Event.Type.Utterance
    df_events = df_events[['type', 'timestamp', 'agent', 'payload']]
    return df_events

def build_episode(df):
    """
    Build an Episode out of a Pandas DataFrame of events. The input DataFrame
    must match the fields of an Event tuple:

    * `type`
    * `timestamp`
    * `agent`
    * `payload`
    
    :param df: A Pandas DataFrame representing events
    :type df: :class:`pandas.DataFrame`
    :return: An Episode containing the given events
    :rtype: :class:`due.episode.Episode`
    """
    df_events = _get_events_dataframe(df)
    agents = set(df_events['agent'])
    starter_agent = df_events.iloc[0].agent
    invited_agent = (agents - set([starter_agent])).pop() # TODO: multiple agents!
    result = Episode(starter_agent, invited_agent)
    result.events = [EventTuple(*e[1:]) for e in df_events.itertuples(name=None)]
    return result
