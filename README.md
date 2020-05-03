# DUE - A framework for Conversational AI

Due is a conversational agent framework built with three goals in mind:

* Episode based, learning oriented: the more you talk, the better it gets
* Action capable: user-defined actions (shell scripts) are natural components of the conversation
* Modular architecture: different Natural Language Understanding (NLU) models can be and integrated in the agents

## Work in progress!
Thanks for stopping by, but be patient: it will take a while before the magic happens... Meanwhile, feel free to get in touch at dario DOT chi AT inventati DOT org, or check on the latest development updates here: https://github.com/dario-chiappetta/Due/projects.

## Setup
**Warning**: this software is not production ready yet, good chanches are it doesn't make sense for you to try it out.

Packaging and dependencies are handled with Poetry (https://python-poetry.org/). Before installing dependencies, make sure the following requirements are satisfied:

* Python 3.7+
* Poetry (see https://python-poetry.org/docs/#installation)
* libmagic (see https://github.com/ahupp/python-magic)

You can now setup the environment for Due as follows:

    poetry install

This will create a virtual environment with all the necessary dependencies. You can add the `--no-dev` option to skip development dependencies, in case you plan on just trying out the software. By default, the virtual environment will not be created under the project's root; if you wish to change this, you can configure Poetry with `poetry config settings.virtualenvs.in-project true`.

Once dependencies are installed, make sure to download Spacy's english models:

    poetry run python -m spacy download en

\[Optional\] If you want to use or train neural models, you will need word embeddings (this part will be moved to a separate repo in next releases):

    mkdir -p ~/.due/resources
    wget http://nlp.stanford.edu/data/glove.6B.zip ~/.due/resources/

## Run

Once the package is installed, you can run a simple agent over XMPP with the following Python code:

```python
# Instantiate an Agent
from due.models.tfidf import TfIdfAgent
agent = TfIdfAgent()

# Learn episodes from a toy corpus
from due.corpora import toy as toy_corpus
agent.learn_episodes(toy_corpus.episodes())
    
# Connect bot
from due.serve import console
console.serve(agent)
```

## Unit testing
This is how you run the unit test suite:

    poetry run pytest

## Documentation
Due is documented with [Sphinx](http://www.sphinx-doc.org). Source documentation files can be found in the `docs/` folder. Docs can be built as follows:

    cd docs/
    poetry run make html

Built HTML files will be placed in the `docs/_build/html` directory. 
