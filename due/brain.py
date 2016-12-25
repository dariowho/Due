from due.episode import Event

from abc import ABCMeta, abstractmethod

import logging


class Brain(metaclass=ABCMeta):
	"""
	A Brain is responsible for storing past episodes, and making good action
	predictions on present ones.
	"""

	def __init__(self, agent):
		self._agent = agent

	@abstractmethod
	def learn_episodes(self, episodes):
		pass

	def learn_episode(self, episode):
		self.learn_episodes([episode])

	@abstractmethod
	def new_episode_callback(self, episode):
		pass

	@abstractmethod
	def utterance_callback(self, episode):
		pass

	@abstractmethod
	def leave_callback(self, episode, agent):
		pass

	@abstractmethod
	def save(self):
		pass

	@staticmethod
	@abstractmethod
	def load(self, saved_brain, agent):
		pass

#
# Cosine Brain (Baseline)
#

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# remove_punctuation_map = dict((ord(c), None) for c in string.punctuation)
_stemmer = nltk.stem.porter.PorterStemmer()
def _normalize(text):
	tokens = nltk.wordpunct_tokenize(text.lower())
	return [_stemmer.stem(t) for t in tokens]

def _cosine_similarity(text, other_text, vectorizer=TfidfVectorizer(tokenizer=_normalize)):
	tfidf = vectorizer.fit_transform([text, other_text])
	return (tfidf*tfidf.T).A[0,1]

class CosineBrain(Brain):
	"""A baseline Brain model that just uses a vetor similarity measure to pick
	appropriate answers to incoming utterances.
	"""
	def __init__(self, agent):
		self._logger = logging.getLogger(__name__ + ".CosineBrain")
		super().__init__(agent)
		self._active_episodes = {}
		self._past_episodes = []

	def learn_episodes(self, episodes):
		self._past_episodes.extend(episodes)

	def new_episode_callback(self, episode):
		self._active_episodes[episode.id] = episode

	def utterance_callback(self, episode):
		answers = self._answers(episode)
		for a in answers:
			if a.type == Event.Type.Utterance:
				self._agent.say(a.payload, episode)
		if len(answers) == 0:
			self._logger.info("No answers found.")

	def leave_callback(self, episode, agent):
		# TODO: should implement some learning logic (eg. learn only if similar 
		#       enough to another episode)
		self.learn_episode(episode)

	def _answers(self, episode):
		last_utterance = episode.last_event(Event.Type.Utterance)
		best_score = 0
		result = []
		for pe in self._past_episodes:
			scores = [_cosine_similarity(last_utterance.payload, u.payload) for u in pe.events if u.type == Event.Type.Utterance]
			max_score = max(scores)
			max_index = scores.index(max_score)
			if max_score > best_score:
				self._logger.info("Best match: '" + pe.events[max_index].payload)
				result = [pe.events[max_index+1]] if max_index < len(pe.events)-1 else result
		return result

	def save(self):
		return {
			'past_episodes': [e.save() for e in self._past_episodes]
		}

	@staticmethod
	def load(saved_brain, agent):
		result = CosineBrain(agent)
		result._past_episodes = saved_brain['past_episodes']
		return result