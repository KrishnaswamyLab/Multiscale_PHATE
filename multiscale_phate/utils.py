import pickle


def hash_object(X):
    return hash(pickle.dumps(X))
