"""
Baseline sentence matching model based on If-Idf vector similarity.
"""
import logging
from datetime import datetime
from collections import namedtuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import due
from due.agent import Agent
from due.event import Event
from due.episode import Episode, extract_utterances
from due.nlp.preprocessing import normalize_sentence

DEFAULT_PARAMETERS = {
	'lemmatize_tokens': False,
}

_UtteranceMetadata = namedtuple('_UtteranceMetadata', ['episode', 'index'])

class TfIdfAgent(Agent):
	"""
	This is a baseline :class:`Agent` that just matches the incoming utterance
	with the ones it has already seen in episodes, and return the one coming
	right after the closest one as an answer.

	Utterance similarity is modeled as the plain **cosine distance** of the
	**tf-idf** sentence vectors.

	The only parameter that can be currently passed to the model is
	`lemmatize_tokens` (defaults to `False`), which adds lemmatization to
	learned utterances.

	:param parameters: A dictionary of parameters.
	:param parameters: `dict`
	:param _data: This is used by :meth:`due.agent.Agent.load`
	:param _data: `dict`
	"""

	def __init__(self, id=None, parameters=None, _data=None):
		parameters = parameters if parameters else {}
		self._logger = logging.getLogger(__name__ + ".TfIdfAgent")
		super().__init__(id)
		self.parameters = {**DEFAULT_PARAMETERS, **parameters} if not _data else {**_data['parameters'], **parameters}
		self._active_episodes = {}
		self._vectorizer = TfidfVectorizer(tokenizer=_dummy_function, preprocessor=_dummy_function)

		self._past_episodes = []
		self._normalized_past_utterances = [] # Sequence of all the utterances in the episodes
		self._vectorized_past_utterances = []
		self._past_utterances_metadata = []   # Per each utterance, remember source episode and position

		if _data:
			self.parameters = _data['parameters']
			self._past_episodes = [Episode.load(e) for e in _data['past_episodes']]
			if self._past_episodes:
				self._normalized_past_utterances = _data['normalized_past_utterances']
				self._past_utterances_metadata = self._load_past_utterances_metadata(_data['past_utterances_metadata'], self._past_episodes)
				self._vectorized_past_utterances = self._vectorizer.fit_transform(self._normalized_past_utterances)

	def learn_episodes(self, episodes):
		"""See :meth:`due.agent.Agent.learn_episodes`"""
		for e in tqdm(episodes):
			self._past_episodes.append(e)
			for i, u in enumerate(extract_utterances(e)):
				if u:
					self._normalized_past_utterances.append(self._process_utterance(u))
					self._past_utterances_metadata.append(_UtteranceMetadata(e, i))
		self._vectorized_past_utterances = self._vectorizer.fit_transform(self._normalized_past_utterances)


	def _process_utterance(self, utterance):
		return normalize_sentence(
			utterance,
			return_tokens=True,
			lemmatize=self.parameters['lemmatize_tokens']
		)

	def action_callback(self, action):
		"""See :meth:`due.agent.Agent.action_callback`"""
		self._logger.debug("Received action: %s", action)

	def utterance_callback(self, episode):
		"""See :meth:`due.agent.Agent.utterance_callback`"""
		last_utterance = episode.last_event(Event.Type.Utterance)
		predicted_answer = self._predict(last_utterance.payload)
		if predicted_answer:
			return [Event(Event.Type.Utterance, datetime.now(), self.id, predicted_answer)]
		return []


	def _predict(self, sentence):
		sentence_v = self._vectorizer.transform([self._process_utterance(sentence)])
		scores = cosine_similarity(self._vectorized_past_utterances, sentence_v)
		max_utterance_meta = self._past_utterances_metadata[np.argmax(scores)]
		matched_past_episode = max_utterance_meta.episode
		matched_index_in_episode = max_utterance_meta.index
		try:
			return matched_past_episode.events[matched_index_in_episode+1].payload
		except IndexError:
			return None

	def new_episode_callback(self, new_episode):
		"""See :meth:`due.agent.Agent.new_episode_callback`"""
		self._logger.debug("New episode callback received: %s", new_episode)

	def leave_callback(self, episode):
		"""See :meth:`due.agent.Agent.leave_callback`"""
		self.learn_episode(episode)

	def save(self):
		"""See :meth:`due.agent.Agent.save`"""
		return {
			'version': due.__version__,
			'class': 'due.models.tfidf.TfIdfAgent',
			'data': {
				'parameters': self.parameters,
				'past_episodes': [e.save() for e in self._past_episodes],
				'normalized_past_utterances': self._normalized_past_utterances,
				'past_utterances_metadata': self._save_past_utterances_metadata()
			}
		}

	def _save_past_utterances_metadata(self):
		result = []
		episode_index = 0
		for pum in self._past_utterances_metadata:
			episode_index = self._past_episode_index(pum.episode, start=episode_index)
			result.append([episode_index, pum.index])
		return result

	def _load_past_utterances_metadata(self, data, past_episodes):
		return [_UtteranceMetadata(past_episodes[x[0]], x[1]) for x in data]

	def _past_episode_index(self, episode, start=0):
		for i in range(start, len(self._past_episodes)):
			if self._past_episodes[i] is episode:
				return i

def _dummy_function(x):
	"""This is used to feed already processed data to TfidfVectorizer"""
	return x
