import logging
# logging.basicConfig(level=logging.INFO)

import pkg_resources

from due.util.resources import ResourceManager

resource_manager = ResourceManager()
with pkg_resources.resource_stream(__name__, 'resource_index.yaml') as f:
	resource_manager.register_yaml(f)

__version__ = '0.1.dev4'
