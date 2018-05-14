"""
This module defines the way resource files (eg. serialized models, corpora...) are
downloaded, cached and retrieved in Due.
"""
import os

DEFAULT_RESOURCE_FOLDER = '~/.due/resources'

class ResourceManager(object):
	"""
	The Resource Manager handles the download, caching and retrieval of resources
	in a project.

	TODO: this class is under development
	"""

	def __init__(self, resource_folder=DEFAULT_RESOURCE_FOLDER):
		self.resource_folder = os.path.expanduser(resource_folder)
		if not os.path.exists(self.resource_folder):
			os.makedirs(self.resource_folder)

	def open_resource(self, name, mode="r"):
		"""
		Return a file object representing the resource with the given name.

		:param name: the name of the resource to open
		:type name: `str`
		:param mode: the mode in which the file is opened
		:type mode: `str`
		"""
		return open(self.resource_path(name), mode)

	def resource_path(self, name):
		"""
		Return the path of the resource with the given name

		:param name: the name of the resource
		:type name: `str`
		"""
		return os.path.join(self.resource_folder, name)
