import unittest

import due

class TestDue(unittest.TestCase):

	def test_resource_manager(self):
		self.assertIsNotNone(due.resource_manager)
		self.assertNotEqual(len(due.resource_manager.resources), 0)