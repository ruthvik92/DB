from functools import wraps
from time import time

global TOTAL_TIME
TOTAL_TIME = {}


def timing(callable):
    @wraps(callable)
    def wrap(*args, **kwargs):
        t1 = time()
        result = callable(*args, **kwargs)
        t2 = time()
        print("Functtion:{} took {} seconds".format(callable.__name__, t2 - t1))
        if callable.__name__ not in TOTAL_TIME.keys():
            TOTAL_TIME[callable.__name__] = []
        TOTAL_TIME[callable.__name__].append(t2 - t1)
        return result

    return wrap
