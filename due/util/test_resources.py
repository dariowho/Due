import unittest
import os
import tempfile
import shutil

from due.util.resources import *

class TestResourceManager(unittest.TestCase):

	def test_register_resource(self):
		with tempfile.TemporaryDirectory() as tmp_dir:
			rm = ResourceManager(resource_folder=tmp_dir)
			rm.register_resource('test.resource', 'a test resource', 'http://fake.com/res.zip', 'test_resource.zip')
			self.assertEqual(rm.resources, {
				'test.resource': ResourceRecord('test.resource', 'a test resource', 'http://fake.com/res.zip', 'test_resource.zip')
			})

			with self.assertRaises(ValueError):
				rm.register_resource('test.resource', 'a duplicate resource', 'http://fake.com/res.zip', 'duplicate_resource.zip')

			with self.assertRaises(ValueError):
				rm.register_resource('test.duplicate.resource', 'a duplicate resource', 'http://fake.com/res.zip', 'test_resource.zip')

	def test_open_resource(self):
		with tempfile.TemporaryDirectory() as tmp_dir:
			content = "this is the file content"
			filename = "test_file.txt"
			with open(os.path.join(tmp_dir, filename), 'w') as fw:
				print(content, file=fw, end='')

			rm = ResourceManager(resource_folder=tmp_dir)
			rm.register_resource('test.resource', 'a test resource', 'http://fake.com/res.zip', filename)

			with rm.open_resource("test.resource") as f:
				self.assertEqual(f.read(), content)

	def test_open_resource_file(self):
		with tempfile.TemporaryDirectory() as tmp_dir:
			zip_tmp_root = os.path.join(tmp_dir, 'zip_tmp_root')
			content = "this is the file content"
			filename = "test_file.txt"
			os.makedirs(zip_tmp_root)
			with open(os.path.join(zip_tmp_root, filename), 'w') as fw:
				print(content, file=fw, end='')
			zipfile_path = os.path.join(tmp_dir, 'test_resource')
			shutil.make_archive(zipfile_path, 'zip', zip_tmp_root)

			rm = ResourceManager(resource_folder=tmp_dir)
			rm.register_resource('test.resource', 'a test ZIP resource', 'http://fake.com/res.zip', 'test_resource.zip')

			with rm.open_resource_file('test.resource', filename) as f:
				self.assertEqual(f.read(), content)

			with rm.open_resource_file('test.resource', filename, binary=True) as f:
				self.assertEqual(f.read(), bytes(content, 'utf-8'))

	def test_error_if_not_found(self):
		with tempfile.TemporaryDirectory() as tmp_dir:
			rm = ResourceManager(resource_folder=tmp_dir)
			rm.register_resource('test.resource', 'a test ZIP resource', 'http://fake.com/res.zip', 'test_resource.zip')
			
			with self.assertRaises(ValueError):
				rm.open_resource('test.resource')

			with self.assertRaises(ValueError):
				rm.open_resource_file('test.resource', 'foo.txt')