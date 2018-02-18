import pickle

def serialize(saved_thing, path, overwrite=True):
	if not overwrite:
		raise NotImplementedError()

	with open(path, 'wb') as f:
		return pickle.dump(saved_thing, f)

def deserialize(path):
	with open(path, 'rb') as f:
		return pickle.load(f)