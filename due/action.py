from abc import ABCMeta, abstractmethod, abstractproperty

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

#
# Example Action
#

import os.path
from datetime import datetime

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