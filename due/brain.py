from abc import ABCMeta, abstractmethod

import logging


class Brain(metaclass=ABCMeta):
	"""
	A Brain is responsible for storing past episodes, and making good action
	predictions on present ones.

	`Brain` is an interface that is meant to be implemented with the greatest
	possible number of models.
	"""

	@abstractmethod
	def learn_episodes(self, episodes):
		"""
		Submit a list of Episodes for the `Brain` to learn. This just wraps calls
		to :meth:`due.brain.Brain.learn_episode`.

		TODO: this should not be an abstract method

		:param episodes: a list of episodes
		:type episodes: `list` of `due.episode.Episode`
		"""
		pass

	def learn_episode(self, episode):
		"""
		Submit an Episode for the Brain to learn.

		After the learning process, the Brain is supposed to provide better answers
		to the incoming Events. Whether this happens depends on the model implementing
		the Brain interface.

		:param episode: an Episode
		:type episode: :class:`due.episode.Episode`
		"""
		self.learn_episodes([episode])

	@abstractmethod
	def new_episode_callback(self, episode):
		"""
		Agents are supposed to call this method to notify their Brain instance
		of the initiation of a new Episode.

		See :meth:`due.agent.Agent.new_episode_callback`

		:param episode: the new Episode
		:type episode: :class:`due.episode.Episode`
		"""
		pass

	@abstractmethod
	def utterance_callback(self, episode):
		"""
		Agents are supposed to call this method to notify their Brain instance
		of a new utterance in an Episode they take part in.

		See :meth:`due.agent.Agent.utterance_callback`

		:param episode: the episode where the new utterance has been posted
		:type episode: :class:`due.episode.Episode`
		"""
		pass

	@abstractmethod
	def leave_callback(self, episode, agent):
		"""
		Agents are supposed to call this method to notify their Brain instance
		when an Agent is leaving an Episode.

		See :meth:`due.agent.Agent.leave_callback`

		:param episode: the Episode where the Agent is leaving
		:type episode: :class:`due.episode.Episode`
		:param agent: the Agent who is leaving
		:type agent: :class:`due.agent.Agent`
		"""
		pass

	@abstractmethod
	def save(self):
		"""
		Saves the Brain to a serializable object that can be reloaded with
		:meth:`due.brain.Brain.load`.

		:return: a serializable representation of `self`
		:rtype: `dict`
		"""
		pass

	@staticmethod
	def load(saved_brain):
		"""
		Loads an object represanting a Brain that was produced by
		:meth:`due.brain.Brain.save`.

		:param saved_brain: the saved Brain
		:type saved_brain: `dict`
		"""
		class_ = dynamic_import(saved_brain['class'])
		return class_(saved_brain['data'])

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
	"""
	A baseline Brain model that just uses cosine distance on TfIdf representations
	of the incoming utterances to pick their appropriate answers.

	see :class:`due.brain.Brain` for details on the available methods.
	"""
	def __init__(self, data=None):
		self._logger = logging.getLogger(__name__ + ".CosineBrain")
		super().__init__()
		self._active_episodes = {}
		self._past_episodes = [Episode.load(e) for e in data['past_episodes']] if data else []

	def learn_episodes(self, episodes):
		self._past_episodes.extend(episodes)

	def new_episode_callback(self, episode):
		self._active_episodes[episode.id] = episode

	def utterance_callback(self, episode):
		answers = self._answers(episode)

		if len(answers) == 0:
			self._logger.info("No answers found.")

		return answers

	def leave_callback(self, episode, agent):
		# TODO: should implement some learning logic (eg. learn only if similar 
		#       enough to another episode)
		self.learn_episode(episode)

	def _answers(self, episode):
		last_utterance = episode.last_event(Event.Type.Utterance)
		self._logger.info("Matching utterance: '%s'" % last_utterance.payload)
		best_score = 0
		best_episode = None
		best_index = None
		result = []
		for pe in self._past_episodes:
			scores = [_cosine_similarity(last_utterance.payload, u.payload) if u.type == Event.Type.Utterance else 0 for u in pe.events]
			max_score = max(scores)
			max_index = scores.index(max_score)
			if max_score > best_score:
				self._logger.info("Best match so far: '%s'. Score: %s" % (pe.events[max_index].payload, max_score))
				# result = [pe.events[max_index+1]] if max_index < len(pe.events)-1 else result
				best_score = max_score
				best_episode = pe
				best_index = max_index

		return CosineBrain._get_answers_to(best_episode, best_index)

	@staticmethod
	def _get_answers_to(episode, index):
		result = []
		if episode is not None and index is not None:
			events = episode.events
			last_index = len(events)-1
			questioning_agent = episode.events[index].agent
			i = index + 1
			# Skip other utterances from same Agent
			while i < last_index and events[i].agent == questioning_agent:
				i += 1
			# Include all replies
			if i < last_index:
				answering_agent = events[i].agent
				while events[i].agent == answering_agent:
					result.append(events[i].clone())
					i += 1
		return result

	def save(self):
		return {
			'class': 'due.brain.CosineBrain',
			'data': {
				'past_episodes': [e.save() for e in self._past_episodes]
			}
		}

# Quick fix for circular dependencies
from due.episode import Episode
from due.event import Event
from due.util import dynamic_import