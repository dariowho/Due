"""
Instantiate a basic agent, train it with a toy corpus, and serve it with a
basic CLI. Usage:

    poetry run python run_example.py

See README.md for instructions on how to setup project dependencies.
"""

if __name__ == "__main__":
    print("Training toy agent...")
    from due.models.tfidf import TfIdfAgent
    agent = TfIdfAgent(id='due')

    # Learn episodes from a toy corpus
    from due.corpora import toy as toy_corpus
    agent.learn_episodes(toy_corpus.episodes())

    print("Welcome! Have a chat with this toy agent, type '!q' to quit.")
    # Connect bot
    from due.serve import console
    console.serve(agent)
