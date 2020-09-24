import time

from caldera.utils.mp import multiprocess


def test_multiprocess_decorate():
    @multiprocess(on="x")
    def foo(x: int, y):
        print(x, y)
        time.sleep(x / 10.0)
        return x, y

    results = foo.pooled(n_cpus=3)(range(10), 3)
    print(results)


def test_multiprocess_decorate_classmethod():
    class Foo:
        @multiprocess(on="x")
        @classmethod
        def foo(cls, x: int, y):
            print(x, y)
            time.sleep(x / 10.0)
            return x, y

    results = Foo.foo.pooled(n_cpus=3)(range(10), 3)
    print(results)
