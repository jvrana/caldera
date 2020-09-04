import random
from itertools import tee

import pytest

from caldera.utils import functional as F


def randint(a, b, n=None):
    if n is None:
        return random.randint(a, b)
    else:
        return [random.randint(a, b) for _ in range(n)]


def validate_generator(generator, expected):
    # assert inspect.isgenerator(generator) or inspect.isgeneratorfunction(generator)

    result_generator, result_generator2 = tee(generator)
    expected_generator, expected_generator2 = tee(expected)
    print()
    print("Result:   " + str(list(result_generator2)))
    print("Expected: " + str(list(expected_generator2)))
    ith = 0
    for _result, _expected in zip(result_generator, expected_generator):
        assert _result == _expected
        ith += 1

    with pytest.raises(StopIteration):
        next(result_generator), "There were values remaining in `result`"
    with pytest.raises(StopIteration):
        next(expected_generator), "There were values remaining in `expected` generator"


class TestParametrizedBasic:
    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "arr", "expected"),
        [
            (
                F.group_each_into_chunks,
                (3,),
                {},
                [1, 2, 3, 4],
                [[1, 2, 3], [4]],
            ),
            (F.group_each_into_chunks, (4,), {}, [1, 2, 3, 4], [[1, 2, 3, 4]]),
            (F.group_each_into_chunks, (5,), {}, [1, 2, 3, 4], [[1, 2, 3, 4]]),
            (
                F.group_each_into_chunks,
                (1,),
                {},
                [1, 2, 3, 4],
                [[1], [2], [3], [4]],
            ),
            (
                F.group_each_into_chunks,
                (3,),
                {},
                range(10),
                [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
            ),
        ],
    )
    def test_chunks(self, func, args, kwargs, arr, expected):
        f = func(*args, **kwargs)
        result = f(arr)
        validate_generator(result, expected)

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "arr", "expected"),
        [
            (
                F.filter_each,
                (lambda x: x >= 3,),
                {},
                range(10),
                list(range(3, 10)),
            ),
            (
                F.filter_each,
                (lambda x: x >= 3,),
                {"inverse": True},
                range(10),
                list(range(3)),
            ),
            (F.filter_each, (lambda x: x == 30,), {}, range(10), []),
        ],
    )
    def test_filter_each(self, func, args, kwargs, arr, expected):
        f = func(*args, **kwargs)
        result = f(arr)
        validate_generator(result, expected)

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "arr", "expected"),
        [
            (
                F.ignore_each_until,
                (lambda x: x == 1,),
                {},
                [10, 2, 1, 2, 3, 1, 4, 2],
                [1, 2, 3, 1, 4, 2],
            ),
        ],
    )
    def test_ignore_until(self, func, args, kwargs, arr, expected):
        f = func(*args, **kwargs)
        result = f(arr)
        validate_generator(result, expected)

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "arr", "expected"),
        [
            (
                F.iter_each_until,
                (lambda x: x == 1,),
                {},
                [10, 2, 3, 1, 2, 3, 1, 4, 2],
                [10, 2, 3],
            ),
        ],
    )
    def test_iter_each_until(self, func, args, kwargs, arr, expected):
        f = func(*args, **kwargs)
        result = f(arr)
        validate_generator(result, expected)

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "arr", "expected"),
        [
            (
                F.ignore_each_count,
                (4,),
                {},
                [10, 2, 3, 1, 2, 3, 1, 4, 2],
                [2, 3, 1, 4, 2],
            ),
        ],
    )
    def test_ignore_each_count(self, func, args, kwargs, arr, expected):
        f = func(*args, **kwargs)
        result = f(arr)
        validate_generator(result, expected)

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "arr", "expected"),
        [
            (
                F.iter_count,
                (4,),
                {},
                [10, 2, 3, 1, 2, 3, 1, 4, 2],
                [10, 2, 3, 1],
            ),
            (
                F.iter_count,
                (40,),
                {},
                [10, 2, 3, 1, 2, 3, 1, 4, 2],
                [10, 2, 3, 1, 2, 3, 1, 4, 2],
            ),
        ],
    )
    def test_iter_count(self, func, args, kwargs, arr, expected):
        f = func(*args, **kwargs)
        result = f(arr)
        validate_generator(result, expected)

    # @pytest.mark.parametrize("n", [0, 1, 10, 30])
    # def test_iter_count_has_remaining(self, n):
    #     data = randint(1, 10, 100)
    #     data_gen = iter(data)
    #     result = F.iter_count(n)(data)
    #     print(data)
    #     validate_generator(result, data[:n])
    #     validate_generator(data_gen, data[n:])

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "arr", "expected"),
        [
            (
                F.iter_step,
                (1,),
                {},
                [10, 2, 3, 1, 2, 3, 1, 4, 2],
                [10, 2, 3, 1, 2, 3, 1, 4, 2][::1],
            ),
            (
                F.iter_step,
                (3,),
                {},
                [10, 2, 3, 1, 2, 3, 1, 4, 2],
                [10, 2, 3, 1, 2, 3, 1, 4, 2][::3],
            ),
            (
                F.iter_step,
                (20,),
                {},
                [10, 2, 3, 1, 2, 3, 1, 4, 2],
                [10, 2, 3, 1, 2, 3, 1, 4, 2][::20],
            ),
        ],
    )
    def test_step(self, func, args, kwargs, arr, expected):
        f = func(*args, **kwargs)
        result = f(arr)
        validate_generator(result, expected)

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "arr", "expected"),
        [
            (
                F.group_each_consecutive,
                (lambda x: x % 2,),
                {},
                [10, 2, 3, 1, 2, 3, 1, 4, 2, 1],
                [[10, 2], [3, 1], [2], [3, 1], [4, 2], [1]],
            )
        ],
    )
    def test_group_each_consecutive(self, func, args, kwargs, arr, expected):
        f = func(*args, **kwargs)
        result = f(arr)
        validate_generator(result, expected)


class TestWithRandomData:
    def test_enumerate_each(self):
        data = randint(1, 1000, 100)
        f = F.enumerate_each()
        result = f(iter(data))
        result = f(iter(data))
        validate_generator(result, enumerate(data))

    def test_enumerate_each_2(self):
        data = randint(1, 1000, 3)
        f = F.enumerate_each()
        list(f(iter(data)))
        result = f(iter(data))
        result = f(iter(data))
        validate_generator(result, enumerate(data))

    def test_side_effect(self):
        data = randint(1, 1000, 100)

        arr = []

        def my_side_effect(a):
            print(a)
            if a < 100:
                arr.append(a)

        f = F.apply_each(my_side_effect)
        result = f(iter(data))
        validate_generator(result, data)
        assert arr == [d for d in data if d < 100]

    def test_index_each(self):
        data = iter([[1, 2, 3], [20, 30]])
        f = F.index_each(0)
        assert list(f(data)) == [1, 20]

    def test_cat(self):
        data1 = randint(1, 10, 100)
        data2 = randint(1, 10, 100)

        f = F.cat(iter(data2))

        result = f(iter(data1))
        print(result)
        assert list(next(result)) == data1
        assert list(next(result)) == data2

        result = f(iter(data1))
        assert list(next(result)) == data1
        assert list(next(result)) == data2

    def test_repeat(self):
        data = iter(range(10))
        f = F.repeat_all(3)
        result = f(data)
        assert list(result) == list(range(10)) * 3

    @pytest.mark.parametrize("n", [0, 1, 50])
    def test_iter_count(self, n):
        data = randint(1, 1000, 50)
        f = F.iter_count(n)
        result = list(f(iter(data)))
        assert len(result) == n
        assert result == data[:n]

    def test_repeat_forever(self):
        data = iter(range(10))
        f = F.compose(F.repeat_all(10), F.iter_count(12))
        result = f(data)
        assert list(result) == (list(range(10)) * 2)[:12]

    def test_pairwise_each(self):
        data = randint(1, 1000, 100)
        expected = zip(data, data[1:])
        result = F.pairwise_each()(iter(data))
        validate_generator(result, expected)

    def test_yield_all(self):
        data = randint(1, 1000, 100)
        result = F.yield_all()(iter(data))
        validate_generator(next(result), data)
        with pytest.raises(StopIteration):
            next(result)

    def test_yield_all_with_func(self):
        def foo(arr):
            yield from range(10)

        data = randint(1, 1000, 100)
        result = F.yield_all(foo)(iter(data))
        validate_generator(next(result), range(10))

    @pytest.mark.parametrize("n", [0, 1, 3, 10])
    def test_tee_all(self, n):
        data = randint(1, 1000, 100)
        result = F.tee_all(n)(iter(data))
        for _ in range(n):
            validate_generator(next(result), iter(data))
        with pytest.raises(StopIteration):
            next(result)

    def test_get_each(self):
        data = randint(1, 1000, 100)
        kv = [{"x": x} for x in data]
        result = F.get_each("x")(iter(kv))
        validate_generator(result, data)

    def test_get_each_with_default(self):
        data = randint(1, 1000, 100)
        kv = [{"x": x} for x in data]
        result = F.get_each("k", default=123)(iter(kv))
        validate_generator(result, [123] * len(data))

    def test_iter_next(self):
        data = randint(1, 1000, 10)
        result = F.iter_next()(iter(data))
        validate_generator(result, data[:1])

    def test_group_each_until(self):

        data = randint(1, 1000, 100)

        f = F.group_each_until(lambda x: len(x) > 5, lambda x: x < 100)

        result = f(iter(data))
        i = 0
        for chunk in result:
            assert all([x < 100 for x in chunk])
            assert len(chunk) <= 6
            i += len(chunk)

    def test_iter_each_unique(self):
        data = randint(1, 100, 1000)
        f = F.iter_each_unique()
        result = f(iter(data))
        validate_generator(iter(sorted(result)), iter(sorted(set(data))))

    def test_iter_each_reverse(self):
        data = randint(1, 100, 1000)
        f = F.iter_reverse()
        result = f(iter(data))
        validate_generator(result, data[::-1])

    def test_zip_each_with(self):
        data = list(range(1000))
        data2 = list(range(1000, 2000))
        data3 = list(range(3000, 4000))
        f = F.zip_each_with(data2, data3)
        result = f(iter(data))
        validate_generator(result, zip(data2, data3, data))

        # ensure result can be obtained again...
        result = f(iter(data))
        validate_generator(result, zip(data2, data3, data))

    def test_zip_each_with_first(self):
        data = list(range(1000))
        data2 = list(range(1000, 2000))
        data3 = list(range(3000, 4000))
        f = F.zip_each_with(data2, data3, first=True)
        result = f(data)
        validate_generator(result, zip(data, data2, data3))

    def test_zipmap_with_each(self):
        data = [0, 10, 3]
        f = F.zipmap_each_with(
            lambda x: x + 1,
            lambda x: x * 2,
            lambda x: x * 3,
        )
        result = f(iter(data))
        assert list(result) == [1, 20, 9]

    def test_reduce_each(self):
        data = list(range(1000))
        data2 = list(range(1000, 2000))
        data3 = list(range(3000, 4000))
        data4 = zip(data, data2, data3)
        expected = [i + (i + 1000) + (i + 3000) for i in data]
        result = F.reduce_each(lambda a, b: a + b)(iter(data4))
        validate_generator(result, expected)


class TestComplexFunctions:
    def test_fn_tee_zip_reduce(self):
        data = iter(range(10))

        piped = F.compose(
            F.tee_all(),
            F.zip_all(),
            F.reduce_all(lambda a, b: a + b),
        )

        print(list(piped(data)))

    def test_try_catch(self):
        data = iter(range(10))

        def raise_if_over_5(x):
            if x > 5:
                raise ValueError
            return x

        piped = F.compose(
            F.map_each(raise_if_over_5),
            F.trycatch(ValueError, catch_yields="opps"),
        )

        for x in piped(data):
            print(x)

    def test_try_catch_fails_to_catch(self):
        data = iter(range(10))

        def raise_if_over_5(x):
            if x > 5:
                raise TypeError
            return x

        piped = F.compose(
            F.map_each(raise_if_over_5),
            F.trycatch(ValueError, catch_yields="opps"),
        )

        with pytest.raises(TypeError):
            for x in piped(data):
                print(x)

    def test_map_all_if(self):
        data = iter(range(10))
        piped = F.compose(
            F.tee_all(),
            F.enumerate_each(),
            F.map_all_if(lambda x: x[0] == 0, lambda x: x[1], lambda x: x[1]),
            F.chain_each(),
        )

        result = piped(data)
        print(list(result))
        for r in result:
            print(list(r))

    def test_tee_pipe(self):
        data = randint(1, 1000, 100)
        f = F.compose(
            F.tee_pipe(
                F.apply_each(print),
                list,
            )
        )
        result = f(iter(data))
        validate_generator(result, data)

    def test_tee_consume(self):
        data = randint(1, 1000, 100)
        f = F.compose(F.tee_consume(F.iter_count(3)))
        result = f(iter(data))
        validate_generator(result, data[3:])

    def test_repeat_raises_type_error(self):
        x = "string"
        with pytest.raises(TypeError):
            F.repeat_all(x)(iter(randint(1, 10, 10)))

    def test_repeat_raises_value_error(self):
        x = -1
        with pytest.raises(ValueError):
            F.repeat_all(x)(iter(randint(1, 10, 10)))

    def test_iter_step_raises_type_error(self):
        x = "string"
        with pytest.raises(TypeError):
            F.iter_step(x)(iter(randint(1, 10, 10)))

    def test_iter_step_raises_value_error(self):
        x = -1
        with pytest.raises(ValueError):
            F.iter_step(x)(iter(randint(1, 10, 10)))

    def test_tee_pipe_yield(self):

        data = range(20)

        f = F.compose(
            F.tee_pipe_yield(F.filter_each(lambda x: x < 2)),
            F.iter_reverse(),
            F.chain_each(),
        )

        assert list(f(data)) == list(range(20)) + [0, 1]
