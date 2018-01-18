def full_class_name(a_class):
	return "%s.%s" % (a_class.__module__, a_class.__name__)

def dynamic_import(name):
	"""https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class"""
	components = name.split('.')
	result = __import__(components[0])
	for c in components[1:]:
		result = getattr(result, c)
	return result