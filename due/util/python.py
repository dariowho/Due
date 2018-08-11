"""
This module contains helper functions related to the Python language.
"""

def full_class_name(a_class):
	"""
	Return the fully qualified name of the given class. Note that `a_class` is
	not a string class name, but an actual class type.

	:param a_class: a Python class
	:type a_class: `type`
	"""
	return "%s.%s" % (a_class.__module__, a_class.__name__)

def dynamic_import(name):
	"""
	Import the given name and return it. Eg.

	>>> p = dynamic_import('os.path')
	>>> type(p)
	module

	Source: https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class

	:param name: an importable name
	:type name: `str`
	:return: the imported entity
	:rtype: *many*
	"""
	components = name.split('.')
	result = __import__(components[0])
	for c in components[1:]:
		result = getattr(result, c)
	return result

def is_notebook():
	"""
	Detect whether `due` is running in a Jupyter Notebook. This is needed
	because TQDM, our progress bar implementation of choice, has
	Jupyter-specific imports.
	"""
	try:
		if type(get_ipython()).__module__.startswith('ipykernel.'):
			return True
	except NameError:
		return False

	return False
