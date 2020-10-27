import pickle


def hash_object(X):
    """Short summary.

    Parameters
    ----------
    X : type
        Description of parameter `X`.

    Returns
    -------
    type
        Description of returned object.

    """
    return hash(pickle.dumps(X))
