from bentham.functional.curry import curry
from bentham.functional.curry import flex_curry


def test_curry():
    def foo(a, b, c):
        return a + b + c

    assert foo(2, 3, 4) == 2 + 3 + 4
    assert curry(foo)(2)(3)(4) == 2 + 3 + 4


def test_flex_curry():
    def foo(a, b, c, d=2):
        return (a, b, c, d)

    f = flex_curry(foo)
    assert f(3, 4, 2) == (3, 4, 2, 1)
    assert f(3)(4, 2) == (3, 4, 2, 1)
    assert f(3)(d=2)(1)(10) == (3, 1, 10, 2)
    assert f(3, 2)(d=2)(30) == (3, 2, 30, 2)
