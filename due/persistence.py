"""
This module defines the way saved entities (see `save` method of `Agent`, `Episode`,
`Brain`, etc) are serialized to files.

Currently, this is simply done with pickle.
"""
import pickle

def serialize(saved_thing, path, overwrite=True):
	"""
	Serialize the given object (return value of the `save` method of the Project
	entities - ie. :class:`due.agent.Agent`, :class:`due.episode.Episode`, ...)
	to a file with the given path.

	:param saved_thing: the saved object
	:type saved thing: *anything serializable*
	:param path: path of the output file
	:type path: `str`
	:param overwrite: overwrite if existing
	:type overwrite: `bool`
	"""
	if not overwrite:
		raise NotImplementedError()

	with open(path, 'wb') as f:
		return pickle.dump(saved_thing, f)

def deserialize(path):
	"""
	Deserialize an object reading from the given path.

	:param path: the path where the object was serialized
	:type path: `str`
	:return: the deserialized object
	:rtype: *many* 
	"""
	with open(path, 'rb') as f:
		return pickle.load(f)