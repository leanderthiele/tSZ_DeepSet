
def MergeDicts(d1, d2) :
    """
    utility function that merges two dicts (not recursive)
    Returns a dict that has all keys from d1 and all keys from d2 that are not in d1.
    """

    assert isinstance(d1, dict) and isinstance(d2, dict)

    out = d1.copy()

    for k, v in d2.items() :
        out.setdefault(k, v)

    return out
