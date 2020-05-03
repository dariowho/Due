"""
API
===
"""
import logging
import sys
from io import StringIO 

class CaptureIO():
	"""
	Capture every kind of screen I/O that happens within the context (adapted from
	this answer: https://stackoverflow.com/a/14058475).

	Example:

	.. code-block:: python

		with CaptureIO() as captured:
			logging.warning("this is root warning")
			print("This is print")
			logging.getLogger('due.serve').warning("this is due warning")

	The above code will produce no screen output. Capture output can be retrieved as follows:

		>>> captured.flush()
		['WARNING:root:this is root warning',
		'This is print',
		'WARNING:due.serve:this is due warning']

	"""
	def __init__(self, logging_only=False):
		self._stringio = StringIO()
		self._stdout = None
		self._stderr = None
		self.log_handlers = None
		self._logging_only = logging_only

	def __enter__(self):
		if not self._logging_only:
			self._stdout = sys.stdout
			self._stderr = sys.stderr
			sys.stdout = self._stringio
			sys.stderr = self._stringio

		root_logger = logging.getLogger()
		self.log_handlers = root_logger.handlers
		default_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		for h in self.log_handlers:
			root_logger.removeHandler(h)
			default_formatter = h.formatter
		handler = logging.StreamHandler(self._stringio)
		handler.setFormatter(default_formatter)
		root_logger.addHandler(handler)

		return self

	def flush(self):
		result = self._stringio.getvalue().splitlines()
		self._stringio.truncate(0)
		return result

	def __exit__(self, *args):
		if not self._logging_only:
			sys.stdout = self._stdout
			sys.stderr = self._stderr

		root_logger = logging.getLogger()
		for h in logging.getLogger().handlers:
			root_logger.removeHandler(h)
		for h in self.log_handlers:
			root_logger.addHandler(h)
