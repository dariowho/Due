# DUE - A learning digital assistant

Due is an autonomous conversational agent built with these goals in mind:

* Learning oriented: the more you talk, the better it gets
* Action capable: user-defined actions (shell scripts) are natural components of the conversation
* Modular architecture: different Natural Language Understanding (NLU) models can be and integrated in the agents

## Work in progress!
Thanks for stopping by, but be patient: it will take a while before the magic happens... Meanwhile, feel free to get in touch at dario DOT chi AT inventati DOT org

## Setup
**Warning**: this software is not production ready yet, good chanches are it doesn't make sense for you to try it out.

If you still want to install it, you can just clone the repo and put it on your `$PYTHONPATH`, or you can build a package as follows:

    $ python setup.py check
    running check
    $ python setup.py sdist

This will produce a `due-0.0.1.tar.gz` package in the `dist/` folder. You can install the package as follows:

    $ cd dist/
    $ pip install due-0.0.1.tar.gz

Once the package is installed, you can run a simple agent over XMPP with the following Python code:

```python
from due.xmpp import DueBot
bot = DueBot("<XMPP_ACCOUNT_USERNAME>", "<XMPP_ACCOUNT_PASSWORD>")

# Learn an example episode
alice = agent.HumanAgent(name="Alice")
bob = agent.HumanAgent(name="Bob")
e1 = alice.start_episode(bob)
alice.say("Hi!", e1)
bob.say("Hi!", e1)
alice.say("How are you?", e1)
bob.say("Good thanks, and you?", e1)
alice.say("I'm doing fine, thank you", e1)
bob.say("Bye", e1)
alice.say("See ya!", e1)
bot.learn_episodes([e1])
    
# Connect bot
bot.connect()
bot.process(block=True)
```

## Unit testing
Unit tests are being written. You can run the test suite as follows:

    $ python setup.py nosetests

## Documentation
*Working on it...*
