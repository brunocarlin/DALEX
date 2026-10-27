"""Keep explicitly seeded explanation streams local without changing MT19937 draws."""
import numpy as np


def get_random_state(seed):
    # Preserve np.random.seed(...) workflows when callers omit random_state.
    # Explicit seeds must never reset another thread's generator.
    return np.random if seed is None else np.random.RandomState(seed)
