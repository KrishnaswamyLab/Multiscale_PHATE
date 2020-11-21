import pickle


def hash_object(X):
    """Compute a unique hash of any Python object.

    Parameters
    ----------
    X : object
        Object for which to compute unique hash

    Returns
    -------
    hash : str
        Unique hash based on pickle dump of X.
    """
    return hash(pickle.dumps(X))
