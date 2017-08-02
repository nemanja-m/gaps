class Cache:
    """Cache for dissimilarity measures of individuals

    Class have static lookp table where keys are Piece's id's.
    For each pair puzzle pieces there is a map with values representing
    dissimilarity measure between them. Each next generation have greater chance to use
    cached value instead of calculating measure again.

    Attributes:
        lookup_table      Dictionary with cached dissimilarity measures for puzzle pieces
        best_match_table  Dictionary with best matching piece for each edge and each piece

        hits    Number of successfully lookups
        misses  Number of unsuccessful lookups

    """

    lookup_table     = {}
    best_match_table = {}

    hits   = 0
    misses = 0

    @classmethod
    def put(cls, ids, orientation, value):
        """Puts a new value in lookup table for given pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'

        :params value:       Value of dissimilarity measure

        Usage::

            >>> from cache import Cache
            >>> Cache.put([1, 2], "TD", 42)

        """

        key = tuple(ids)

        if not cls.contains(ids):
            cls.lookup_table[key] = {}

        cls.lookup_table[key][orientation] = value
        cls.misses += 1

    @classmethod
    def get(cls, ids, orientation):
        """Returns previously cached dissimilarity measure for input pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'

        Usage::

            >>> from cache import Cache
            >>> Cache.get([1, 2], "TD")
            >>> 42

        """

        key        = tuple(ids)
        cls.hits  += 1
        return cls.lookup_table[key][orientation]

    @classmethod
    def contains(cls, ids, orientation=None):
        """Check if there are any dissimilarity measures cached for given pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'
                             Default is None

        Usage::

            >>> from cache import Cache
            >>> Cache.contains([1, 2], "TD")
            >>> True

        """

        key = tuple(ids)

        if key in cls.lookup_table:
            if orientation != None:
                return orientation in cls.lookup_table[key]
            else:
                return True
        else:
            return False

    @classmethod
    def best_match(cls, piece, orientation):
        return cls.best_match_table[piece][orientation][0][0]
