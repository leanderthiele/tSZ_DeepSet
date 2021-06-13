
class FixedLenVec(type) :
    """
    metaclass that we can use for classes that are constructed as vectors
    of fixed length
    """

    def __len__(self) :
        return self._length()
