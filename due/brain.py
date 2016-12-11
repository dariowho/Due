class Brain(metaclass=ABCMeta):
	"""
	A Brain is responsible for storing past episodes, and making good action
	predictions on present ones.
	"""

	def __init__(self):
		pass

	@abstractmethod
	def record(self, episode):
		pass