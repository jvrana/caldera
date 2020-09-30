import inspect
import re
from multiprocessing.pool import IMapIterator
from multiprocessing.pool import Pool
from typing import Callable
from typing import List
from typing import Tuple
from typing import TypeVar

import dill

_T = TypeVar("T")
_S = TypeVar("S")
_Arg = TypeVar("A")


def run_payload(payload: bytes) -> Tuple[int, _T]:
    unloaded: Tuple[Callable[[_Arg], _T], _Arg, int] = dill.loads(payload)
    fun, args, idx = unloaded
    return idx, fun(args)


def create_payloads(func: Callable[[_Arg], _T], args: List[_Arg]) -> List[bytes]:
    payloads = []
    for idx, arg in enumerate(args):
        data = (func, arg, idx)
        payloads.append(dill.dumps(data))
    return payloads


def handle_result(
    results: IMapIterator, callback=None, error_callback=None
) -> List[Tuple[int, _T]]:
    results_list = []
    while True:
        r = ...
        try:
            idx, r = next(results)
        except StopIteration:
            break
        except Exception as e:
            if error_callback:
                error_callback(e)
            else:
                raise e
        finally:
            if r is not ...:
                if callback:
                    callback(r)
                results_list.append((idx, r))
    return results_list


def _identity_wrapper(f: Callable[[_S], _T]) -> Callable[[_Arg], _T]:
    return f


def star_wrapper(f: Callable[[_S], _T]) -> Callable[[_Arg], _T]:
    def wrapped(args):
        return f(*args)

    return wrapped


def _resolved_sorted_handler(resolved: List[Tuple[int, _T]]) -> List[_T]:
    return [r[1] for r in sorted(resolved)]


_default_wrapper = _identity_wrapper
_default_resolved_handler = _identity_wrapper


def run_with_pool(
    pool: Pool,
    func: Callable,
    args: List[_Arg],
    wrapper: Callable[[Callable[[_S], _T]], Callable[[_Arg], _T]] = None,
    chunksize: int = 1,
    callback: Callable[[_T], None] = None,
    error_callback: Callable[[Exception], None] = None,
    resolved_handler: Callable[[int, _T], _S] = None,
) -> _S:
    wrapper = wrapper or _default_wrapper
    payloads = create_payloads(wrapper(func), args)
    results = pool.imap_unordered(run_payload, payloads, chunksize=chunksize)
    resolved: List[Tuple[int, _T]] = handle_result(results, callback, error_callback)

    resolved_handler = resolved_handler or _default_resolved_handler
    return resolved_handler(resolved)


def argmap(f, args, kwargs):
    argspec = inspect.getfullargspec(f)
    mapping = {}
    for arg, val in zip(argspec.args, args):
        mapping[arg] = val
    mapping.update(kwargs)
    return mapping


_varnamepattern = re.compile(r"[a-zA-Z_][\w\d]*")


def valid_varname(x: str):
    if not isinstance(x, str):
        return False
    if _varnamepattern.match(x) is not None:
        return True
    return False
