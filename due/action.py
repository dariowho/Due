from abc import ABCMeta, abstractmethod, abstractproperty
from due.util.python import full_class_name, dynamic_import

#
# Action Interface
#

class Action(metaclass=ABCMeta):
	"""
	An Action is an Event in an Episode that allows the run of arbitrary Python
	code.

	This is achieved extending this `Action` interface with the proper implementation
	of the `run` method.
	"""

	@abstractmethod
	def run(self):
		"""
		This method will be invoked when the Agent issues the Action in a
		conversation
		"""
		pass

	def save(self):
		"""
		Save the Action as a serializable object.

		:return: the serialized Action
		:rtype: `dict`
		"""
		return {'class': full_class_name(self.__class__), 'data': None}

	@staticmethod
	def load(saved_action):
		"""
		Load an action from an object produced with :func:`due.action.Action.save`

		:param saved_action: the saved action
		:type saved_action: `dict`
		:return: an `Action` object
		:rtype: :class:`due.action.Action`
		"""
		class_ = dynamic_import(saved_action['class'])
		return class_(saved_action['data'])

#
# Example Actions
#

import os.path
from datetime import datetime

class RecordedAction(Action):
	"""
	Example action that just stores a boolean `done` variable which is set to True
	when the Action is run.

	:param data: data from another action (this is only used by `load`)
	:type data: `dict`
	"""

	def __init__(self, data=None):
		self.done = data['done'] if data else False

	def run(self):
		"""
		Runs the Action by setting `self.done` to `True`
		"""
		self.done = True
		return True

	def save(self):
		"""See :func:`due.action.Action.save`"""
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
		"""
		Runs the Action by creating a file named `DUE_EXAMPLE_ACTION` in your
		home directory.
		"""
		home = os.path.expanduser("~")
		with open(os.path.join(home, ExampleAction.FILENAME), "w") as f:
			f.write(str(datetime.now()))
			f.write("\n")
			f.write(ExampleAction.CONTENT)
		return True
