import time
from multiprocessing import Pool

from caldera.utils.mp import run_with_pool

N_JOBS = 2


def external_func1(*args, **kwargs):
    time.sleep(0.01)
    return (args, kwargs)


def test_default_pool():
    with Pool() as pool:
        results = run_with_pool(pool, external_func1, range(10))
    expected = [(i, ((i,), {})) for i in range(10)]
    t = lambda x: sorted(list(map(str, x)))
    assert t(results) == t(expected)


def test_remove_idx():
    def resolved_handler(r):
        return [_r[1] for _r in r]

    with Pool(N_JOBS) as pool:
        results = run_with_pool(
            pool, external_func1, range(10), resolved_handler=resolved_handler
        )
    expected = [((i,), {}) for i in range(10)]
    t = lambda x: sorted(list(map(str, x)))
    assert t(results) == t(expected)


def test_star_handler():
    def resolved_handler(r):
        return [_r[1] for _r in r]

    def star(f):
        def wrapped(args):
            return f(*args)

        return wrapped

    with Pool(N_JOBS) as pool:
        results = run_with_pool(
            pool,
            external_func1,
            ((i,) for i in range(10)),
            resolved_handler=resolved_handler,
            wrapper=star,
        )
    expected = [((i,), {}) for i in range(10)]
    t = lambda x: sorted(list(map(str, x)))
    assert t(results) == t(expected)


def test_internally_defined_func_default_pool():
    def internal_func(*args, **kwargs):
        return args, kwargs

    with Pool(N_JOBS) as pool:
        results = run_with_pool(pool, internal_func, range(10))

    expected = [(i, ((i,), {})) for i in range(10)]
    t = lambda x: sorted(list(map(str, x)))
    assert t(results) == t(expected)
