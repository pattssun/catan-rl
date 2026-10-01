import random
from contextlib import contextmanager


@contextmanager
def isolated_random(rng):
    # Catanatron uses Python's global RNG even when executing copied games.
    engine_state = random.getstate()
    random.setstate(rng.getstate())
    try:
        yield
    finally:
        rng.setstate(random.getstate())
        random.setstate(engine_state)
