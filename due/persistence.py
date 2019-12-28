"""
This module defines the way saved entities (see `save` method of `Agent`,
`Episode`, `Brain`, etc) are serialized to files.

Currently, this is simply done with **YAML** or JSON.

**Pickle** is also supported, even though it's not advisable to distribute `.pkl`
files, because the format is inherently unsafe
(https://docs.python.org/3/library/pickle.html).

API
===
"""
import json
import pickle

import yaml
import magic

def serialize(saved_thing, path, file_format='yaml', overwrite=True):
	"""
	Serialize the given object (return value of the `save` method of the Project
	entities - ie. :class:`due.agent.Agent`, :class:`due.episode.Episode`, ...)
	to a file with the given path.

	The supported output formats are:

	* `yaml` (default)
	* `json`
	* `pickle`

	:param saved_thing: the saved object
	:type saved_thing: *anything serializable*
	:param path: path of the output file
	:type path: `str`
	:param file_format: 'yaml', 'json' or 'pickle'
	:type file_format: `str`
	:param overwrite: overwrite if existing
	:type overwrite: `bool`
	"""
	if not overwrite:
		raise NotImplementedError()

	if file_format == 'yaml':
		with open(path, 'w') as f:
			yaml.dump(saved_thing, f)
	elif file_format == 'json':
		with open(path, 'w') as f:
			json.dump(saved_thing, f)
	elif file_format == 'pickle':
		with open(path, 'wb') as f:
			return pickle.dump(saved_thing, f)
	else:
		raise ValueError(f"Unsupported file format '{file_format}'. Supported types are 'yaml', 'json' or 'pickle'")

def deserialize(path, file_format='auto', allow_pickle=False):
	"""
	Deserialize an object reading from the given path.

	Supported formats are YAML, JSON and Pickle. Note that, as Pickle is not a
	safe format for public distribution, its support must be explicitly enabled
	with the `allow_pickle` flag. If you do not trust your sources, leave the
	flag to `False` (see https://docs.python.org/3/library/pickle.html).

	:param path: the path where the object was serialized
	:type path: `str`
	:param file_format: if not 'auto', force a specific format (currently unsupported)
	:type file_format: `str`
	:param allow_pickle: Allow reading Pickle binary files
	:type allow_pickle: `bool`
	:return: the deserialized object
	:rtype: *many*
	"""
	if file_format != 'auto':
		raise NotImplementedError()

	detected_format = magic.Magic().from_file(path)

	if detected_format == 'data':
		if not allow_pickle:
			raise ValueError("Binary file detected, but 'allow_pickle' is False. Aborting.")
		with open(path, 'rb') as f:
			return pickle.load(f)
	else:
		with open(path, 'r') as f:
			return yaml.load(f, Loader=yaml.FullLoader)
