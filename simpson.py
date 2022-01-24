#!/usr/bin/env python

# Simpson Diversity Index
# http://en.wikipedia.org/wiki/Diversity_index

# modified from Shannon Diversity Index implementation by audy
# https://gist.github.com/audy/783125
# https://gist.github.com/audy

def simpson_di(data):

    """ Given a hash { 'species': count } , returns the Simpson Diversity Index
    
    >>> simpson_di({'a': 10, 'b': 20, 'c': 30,})
    0.3888888888888889
    """

    def p(n, N):
        """ Relative abundance """
        if n ==  0:
            return 0
        else:
            return float(n)/N

    N = sum(data.values())

    return sum(p(n, N)**2 for n in data.values() if n != 0)


def inverse_simpson_di(data):
    """ Given a hash { 'species': count } , returns the inverse Simpson Diversity Index
    
    >>> inverse_simpson_di({'a': 10, 'b': 20, 'c': 30,})
    2.571428571428571
    """
    return float(1)/simpson_di(data)