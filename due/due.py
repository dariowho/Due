"""
Due is a learning, modular, action-oriented digital assistant.
"""



class Due(object):
	"""
	Main entry point for Due. Should be instantiated with a Brain
	"""

	def __init__(self, brain):
		self._brain = brain
