import re
from functools import lru_cache

import spacy

def tokenize_sentence(sentence, language):
	"""
	Wraps around Spacy's tokenizer returning a list of string tokens for the
	given sentence.

	:param sentence: a sentence
	:type sentence: `str`
	:param language: An ISO 639-1 language code ('en', 'it', ...)
	:type language: `str`
	"""
	s_spacy = _load_spacy(language)(sentence)
	return [str(token) for token in s_spacy]

def normalize_sentence(sentence, return_tokens=False, language='en'):
	"""
	Return a normalized version of the input sentence. Normalization is
	currently limited to:

	* Tokenization
	* Lowercase transformation

	Optionally, a list of tokens can be returned instead of a whole normalized
	string.

	:param sentence: a sentence
	:type sentence: `str`
	:param return_tokens: whether to return a list of `str` tokens or a whole string
	:param language: An ISO 639-1 language code ('en', 'it', ...)
	:type language: `str`
	:return: a normalized sentence
	:rtype: `str` or (`list` of `str`)
	"""
	result = sentence.lower()
	result = re.sub(r'\s+', ' ', result)
	result = tokenize_sentence(result, language)
	if not return_tokens:
		result = ' '.join(result)
	return result

@lru_cache(8)
def _load_spacy(language):
	return spacy.load('en')
