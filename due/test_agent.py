import unittest

from due.agent import *
from due.action import Action

class RecordedAction(Action):
	def __init__(self):
		self.done = False

	def run(self):
		self.done = True
		return True

class TestAgent(unittest.TestCase):

	def test_agent_episode(self):
		ha_alice = HumanAgent(name='Alice')
		ha_bob = HumanAgent(name='Bob')
		e = ha_alice.start_episode(ha_bob)
		self.assertEqual(len(e.events), 0)

		ha_alice.say('alice1', e)
		self.assertEqual(len(e.events), 1)
		self.assertEqual(e.events[-1].payload, 'alice1')

		ha_alice.say('alice2', e)
		self.assertEqual(len(e.events), 2)
		self.assertEqual(e.events[-1].payload, 'alice2')

		ha_bob.say('bob1', e)
		self.assertEqual(len(e.events), 3)
		self.assertEqual(e.events[-1].payload, 'bob1')

		action = RecordedAction()
		ha_alice.do(action, e)
		self.assertTrue(action.done)
		self.assertEqual(len(e.events), 4)
		self.assertTrue(e.events[-1].payload is action)

		ha_alice.leave(e)


	def test_human_agent_load_save(self):
		ha_alice = HumanAgent(name='Alice')
		saved_alice = ha_alice.save()

		loaded_alice = HumanAgent.load(saved_alice)
		self.assertEqual(ha_alice.id, loaded_alice.id)
		self.assertEqual(ha_alice.name, loaded_alice.name)

	def test_due_agent_load_save(self):
		due = Due()
		saved_due = due.save()

		loaded_due = Due.load(saved_due)
		self.assertEqual(due.id, loaded_due.id)
		self.assertEqual(due.name, loaded_due.name)