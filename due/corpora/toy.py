"""
Load a toy corpus that can be used for quick test and development of Due. The
corpus consists of a small set of hand-crafted smalltalk-ish Episodes between
two Agents.
"""
import os
import tempfile
try:
    import importlib.resources as importlib_resources
except ImportError:
    import importlib_resources

from due import corpora
from due.episode import Episode
from due.persistence import deserialize

def episodes():
    # TODO: add support for file buffers in `deserialize`
    toy_yaml = importlib_resources.read_text(corpora, 'toy.yaml')
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, 'toy.yaml')
        with open(path, 'w') as f:
            f.write(toy_yaml)
        saved_episodes = deserialize(path)

    for e in saved_episodes:
        yield Episode.load(e)
