from abc import ABCMeta, abstractmethod, abstractproperty
from due.util import full_class_name, dynamic_import

#
# Action Interface
#

class Action(metaclass=ABCMeta):

	@abstractmethod
	def run(self):
		"""
		This method will be invoked when the Agent issues the Action in a
		conversation
		"""
		pass

	def save(self):
		return {'class': full_class_name(self.__class__), 'data': None}

	@staticmethod
	def load(saved_action):
		class_ = dynamic_import(saved_action['class'])
		return class_(saved_action['data'])

#
# Example Actions
#

import os.path
from datetime import datetime

class RecordedAction(Action):
	def __init__(self, data=None):
		self.done = data['done'] if data else False

	def run(self):
		self.done = True
		return True

	def save(self):
		return {'class': full_class_name(self.__class__), 'data': {'done': self.done}}

	def __eq__(self, other):
		return self.done == other.done

class ExampleAction(Action):
	"""
	An example action that creates a file in your home directory.
	"""

	FILENAME = "DUE_EXAMPLE_ACTION"
	CONTENT = "Due was here..."

	def run(self):
		home = os.path.expanduser("~")
		with open(os.path.join(home, ExampleAction.FILENAME), "w") as f:
			f.write(str(datetime.now()))
			f.write("\n")
			f.write(ExampleAction.CONTENT)
		return True