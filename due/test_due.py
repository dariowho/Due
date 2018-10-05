import unittest

import due
from due.due import Due

class TestDue(unittest.TestCase):

	def test_resource_manager(self):
		self.assertIsNotNone(due.resource_manager)
		self.assertNotEqual(len(due.resource_manager.resources), 0)

	def test_due_agent_load_save(self):
		original_due = Due()
		saved_due = original_due.save()

		loaded_due = Due.load(saved_due)
		self.assertEqual(original_due.id, loaded_due.id)
		self.assertEqual(original_due.name, loaded_due.name)
