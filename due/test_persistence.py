import os
import tempfile

import pytest

from due.persistence import serialize, deserialize

TEST_OBJECT = {
    'a': 1,
    'b': {
        'foo': [1, 2, 3],
        'bar': 42
    },
    'c': [{
        "1": 2,
        "3": [4, 5, 6]
    }, {
        'foo': 'bar',
        'bar': 'foo'
    }]
}

class TestPersistence(object):

    def test_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'saved.yaml')
            serialize(TEST_OBJECT, path, file_format='yaml')
            assert deserialize(path) == TEST_OBJECT

    def test_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'saved.json')
            serialize(TEST_OBJECT, path, file_format='json')
            assert deserialize(path) == TEST_OBJECT

    def test_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'saved.pickle')
            serialize(TEST_OBJECT, path, file_format='pickle')
            assert deserialize(path, allow_pickle=True) == TEST_OBJECT

    def test_pickle_no_flag(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'saved.pickle')
            serialize(TEST_OBJECT, path, file_format='pickle')
            with pytest.raises(ValueError):
                deserialize(path)
