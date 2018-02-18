import unittest

from due.util import *

class TestUtil(unittest.TestCase):

	def test_full_class_name(self):
		from due.agent import Agent
		self.assertEqual(full_class_name(Agent), 'due.agent.Agent')
		
	def test_dynamic_import(self):
		from due.agent import Agent
		DynamicallyImportedAgent = dynamic_import('due.agent.Agent')
		self.assertEqual(DynamicallyImportedAgent, Agent)