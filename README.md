# DUE - An Episodic framework for Conversational AI

Due is a framework for conversational agents that is built with three goals in mind:

* Episode based, learning oriented: the more you talk, the better it gets
* Action capable: user-defined actions (shell scripts) are natural components of the conversation
* Modular architecture: different Natural Language Understanding (NLU) models can be and integrated in the agents

## Work in progress!
Thanks for stopping by, but be patient: it will take a while before the magic happens... Meanwhile, feel free to get in touch at dario DOT chi AT inventati DOT org.

## Setup
Packaging and dependencies are managed by Poetry (https://python-poetry.org/). Before installing dependencies, make sure the following requirements are satisfied:

* Python 3.7+
* Poetry (see https://python-poetry.org/docs/#installation)
* libmagic (see https://github.com/ahupp/python-magic)

You can now setup the environment for Due as follows (this will install dependencies in a virtualenv. Tip: if you wish to create the virtualenv inside the project folder, you can tell Poetry with `poetry config virtualenvs.in-project true`)

    poetry install

Once dependencies are installed, make sure to download Spacy's English models

    poetry run python -m spacy download en

## Run

Once the package is installed, you can run a simple agent with the following Python code:

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
