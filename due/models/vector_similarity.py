import logging
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from due.brain import Brain
from due.event import Event
from due.episode import Episode, extract_utterances
from due.nlp.preprocessing import normalize_sentence

DEFAULT_PARAMETERS = {
	'lemmatize_tokens': False,
}

class TfIdfCosineBrain(Brain):

	def __init__(self, parameters={}, data=None):
		self._logger = logging.getLogger(__name__ + ".VectorSimilarityBrain")
		super().__init__()
		self.parameters = {**DEFAULT_PARAMETERS, **parameters} if not data else {**data['parameters'], **parameters}
		self._active_episodes = {}
		self._vectorizer = TfidfVectorizer(tokenizer=_dummy_function, preprocessor=_dummy_function)

		self._past_episodes = []
		self._normalized_past_episodes = []
		self._vectorized_past_episodes = []

		if data:
			past_episodes = [Episode.load(e) for e in data['past_episodes']]
			self.learn_episodes(past_episodes)

	def learn_episodes(self, episodes):
		self._past_episodes.extend(episodes)

		for e in tqdm(episodes):
			new_utterances = [self._process_utterance(u) if u else None for u in extract_utterances(e)]
			self._past_episodes.append(e)
			self._normalized_past_episodes.append(new_utterances)

		self._vectorizer.fit([x for e in self._normalized_past_episodes for x in e if x])
		self._vectorized_past_episodes = self._vectorize_past_episodes()


	def _process_utterance(self, utterance):
		return normalize_sentence(
			utterance,
			return_tokens=True,
			lemmatize=self.parameters['lemmatize_tokens']
		)

	def _vectorize_past_episodes(self):
		"""
		Makes a list of lists of vectors
		"""
		result = []
		for e in self._normalized_past_episodes:
			result.append([self._vectorizer.transform([s]) if s else None for s in e])
			# result.append([np.asarray(self._vectorizer.transform([s])).squeeze() for s in e])
		return result

	def utterance_callback(self, episode):
		last_utterance = episode.last_event(Event.Type.Utterance)
		predicted_answer = self._predict(last_utterance.payload)
		# TODO: add Agent ID to returned Event
		return [Event(Event.Type.Utterance, datetime.now(), None, predicted_answer)] if predicted_answer else []


	def _predict(self, sentence):
		sentence_v = self._vectorizer.transform([self._process_utterance(sentence)])
		scores = [[cosine_similarity(sentence_v, u)[0][0] if u is not None else 0 for u in episode] for episode in self._vectorized_past_episodes]
		max_episode_index, max_episode_scores = max(enumerate(scores), key=lambda x: max(x[1]))
		max_utterance_index = np.argmax(np.array(max_episode_scores))
		try:
			return self._past_episodes[max_episode_index].events[max_utterance_index+1].payload
		except IndexError:
			return None

	def new_episode_callback(self, episode):
		self._logger.debug("New episode callback received: %s", episode)

	def leave_callback(self, episode):
		# TODO: separate from gold standard episodes
		self.learn_episode(episode)

	def save(self):
		# TODO: save processed sentences to improve loading speed
		return {
			'class': 'due.models.vector_similarity.TfIdfCosineBrain',
			'data': {
				'parameters': self.parameters,
				'past_episodes': [e.save() for e in self._past_episodes]
			}
		}

def _dummy_function(x):
	"""This is used to feed already processed data to TfidfVectorizer"""
	return x