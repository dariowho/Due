"""
This module defines the way resource files (eg. serialized models, corpora...) are
downloaded, cached and retrieved in Due.
"""
import os
import logging
from io import TextIOWrapper
from zipfile import ZipFile
from collections import namedtuple

import yaml
from magic import Magic

DEFAULT_RESOURCE_FOLDER = '~/.due/resources'

ResourceRecord = namedtuple('ResourceRecord', ['name', 'description', 'url', 'filename'])

class ResourceManager(object):
	"""
	The Resource Manager handles the download, caching and retrieval of resources
	in a project.
	"""

	def __init__(self, resource_folder=DEFAULT_RESOURCE_FOLDER):
		self._logger = logging.getLogger(__name__)
		self.resource_folder = os.path.expanduser(resource_folder)
		if not os.path.exists(self.resource_folder):
			os.makedirs(self.resource_folder)
		self.resources = {}
		self.resource_filenames = {}

	def register_resource(self, name, description, url, filename):
		"""
		Register a new resource in the Resource Manager. Once a resource is registered,
		other modules in the application can access it by its name.

		:param name: the name of the new resource (eg. "corpora.cornell")
		:type name: `str`
		:param description: a text description of the resource file
		:type description: `str`
		:param url: URL to download the resource, if missing
		:type url: `str`
		:param filename: name by which the resource file will be looked up in the resources folder
		"""
		if name in self.resources:
			self._logger.error("Entry '%s' already exist in ResourceManager.", name)
			raise ValueError("Cannot overwrite existing resource '%s'" % name)

		if filename in self.resource_filenames:
			resource = self.resource_filenames[filename]
			self._logger.error("Entry '%s' exists in ResourceManager with the same filename (%s).", resource, filename)
			raise ValueError("Cannot overwrite existing resource '%s'" % name)

		self.resources[name] = ResourceRecord(name, description, url, filename)
		self.resource_filenames[filename] = name

	def register_yaml(self, yaml_stream):
		"""
		Read a YAML file containing an index of resources. Such a file must contain a
		`resources` key, having a list of objects as values. These objects must contain
		the `name`, `description`, `url` and `filename` keys (see :meth:`register_resource`
		for an explanation of these values).

		This is an example content for a YAML file specifying one resource to register:

		.. code-block:: yaml

			resources:
			- name: cornell
				description: the Cornell Movie-Dialogs Corpus (http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
				url: http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
				filename: cornell_movie_dialogs_corpus.zip

		:param path: pointer to the YAML file
		:type path: `file object`
		"""
		resource_list = yaml.load(yaml_stream, Loader=yaml.FullLoader)['resources']

		for r in resource_list:
			self.register_resource(r['name'], r['description'], r['url'], r['filename'])

	def open_resource(self, name, mode="r"):
		"""
		Return a file object representing the resource with the given name.

		:param name: the name of the resource to open
		:type name: `str`
		:param mode: the mode in which the file is opened
		:type mode: `str`
		"""
		self._error_if_not_found(name)
		
		return open(self.resource_path(name), mode)

	def open_resource_file(self, name, filename, binary=False, encoding='utf-8'):
		"""
		If the given resource is a compressed archive, extract the given filename
		and return a file pointer to the extracted file.

		Currently, only **ZIP** files are supported.

		:param name: the name of the resource containing the file
		:type name: `str`
		:param filename: the name of the file to extract and return
		:type filename: `str`
		:param binary: if True, open the file in binary mode ("rb")
		"""
		self._error_if_not_found(name)

		path = self.resource_path(name)
		magic = Magic(mime=True)
		mime = magic.from_file(path)
		if mime != 'application/zip':
			raise ValueError("Unsupported MIME type: %s (application/zip is required)" % mime)

		zipfile = ZipFile(path)
		f =  zipfile.open(filename, mode='r')

		return f if binary else TextIOWrapper(f, encoding) # TODO: test encoding

	def resource_path(self, name):
		"""
		Return the path of the resource with the given name

		:param name: the name of the resource
		:type name: `str`
		"""
		return os.path.join(self.resource_folder, self.resources[name].filename)

	def _error_if_not_found(self, name):
		record = self.resources[name]
		path = self.resource_path(name)
		if not os.path.isfile(path):
			print( ("Couldn't find resource '%s'. Download the file at %s and "
				    "copy it in your resource folder (%s) with name '%s' to make "
				    "it available in Due.") % (name, record.url, self.resource_folder, record.filename))
			raise ValueError("Resource not found: %s. Please refer to the logs for guidance." % name)
