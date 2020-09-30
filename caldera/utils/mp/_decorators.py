import inspect
from functools import wraps
from multiprocessing import Pool
from typing import Callable
from typing import List
from typing import Optional

from ._mp_tools import _resolved_sorted_handler
from ._mp_tools import _S
from ._mp_tools import _T
from ._mp_tools import argmap
from ._mp_tools import run_with_pool
from ._mp_tools import valid_varname

# class MultiProcess(object):
#
#     def __init__(self, on: str, auto_annotate: bool = True, attach: bool = True, attach_as="pooled"):
#         if not isinstance(attach_as, str):
#             raise TypeError("attach_as must be a str, not a {}".format(attach_as.__class__))
#         if not valid_varname(attach_as):
#             raise ValueError("attach_as '{}' is not a valid variable name".format(attach_as))
#
#
#         self.on = on
#         self.auto_annotate = auto_annotate
#         self.attach = attach
#         self.attach_as = attach_as
#         self.validate()
#
#     def validate(self):
#         self.validate_on()
#         self.validate_attach_as()
#
#     def validate_attach_as(self):
#         if not isinstance(self.attach_as, str):
#             raise TypeError("attach_as must be a str, not a {}".format(self.attach_as.__class__))
#         if not valid_varname(self.attach_as):
#             raise ValueError("attach_as '{}' is not a valid variable name".format(self.attach_as))
#
#     def validate_on(self):
#         if not isinstance(self.on, str):
#             raise TypeError('argument must be a str, found a {} {}'.format(self.on, self.on.__class__))
#         elif not valid_varname(self.on):
#             raise ValueError("argument '{}' is not a valid variable name".format(self.on))
#
#     def validate_func(self, f):
#         argspec = inspect.getfullargspec(f)
#         if not self.on in argspec.args:
#             raise ValueError("argument '{}' not in signature. Use any of {}".format(self.on, argspec.args))
#
#     @staticmethod
#     def starstar(f):
#         def wrapped(kwargs):
#             return f(**kwargs)
#         return wrapped
#
#     def pooled_inner(self, f, pool_kwargs, override_kwargs_key: str):
#         @wraps(f)
#         def _pooled_inner(*args, **kwargs) -> List[_S]:
#             inner_kwargs = dict(pool_kwargs)
#             if override_kwargs_key is not None:
#                 if override_kwargs_key in kwargs:
#                     pooled_opts = kwargs[override_kwargs_key]
#                     inner_kwargs.update(pooled_opts)
#             inner_kwargs = dict(pool_kwargs)
#             n_cpus = inner_kwargs['n_cpus']
#             chunksize = inner_kwargs['chunksize']
#             callback = inner_kwargs['callback']
#             error_callback = inner_kwargs['error_callback']
#
#             amap = argmap(f, args, kwargs)
#             pooled_kwargs = []
#             if not hasattr(amap[self.on], '__iter__'):
#                 raise ValueError('argument "{}" is not iterable for pooled function'.format(on))
#             for v in amap[self.on]:
#                 _kwargs = dict(amap)
#                 _kwargs[self.on] = v
#                 pooled_kwargs.append(_kwargs)
#
#             with Pool(n_cpus) as pool:
#                 run_with_pool(pool,
#                               f,
#                               wrapper=self.starstar,
#                               chunksize=chunksize,
#                               args=pooled_kwargs,
#                               callback=callback,
#                               error_callback=error_callback,
#                               resolved_handler=_resolved_sorted_handler
#                               )
#         return _pooled_inner
#
#     # @staticmethod
#     # def override(f, outer_kwargs: Dict[str, Any], override_kwargs: str):
#     #     @wraps(f)
#     #     def wrapped(*args, **kwargs):
#     #         inner_kwargs = dict(outer_kwargs)
#     #         if override_kwargs is not None:
#     #             if override_kwargs in kwargs:
#     #                 pooled_opts = kwargs[override_kwargs]
#     #                 inner_kwargs.update(pooled_opts)
#     #         f(*args, **inner_kwargs)
#     #     return wrapped
#
#     def do_auto_annotate(self, func, type = List):
#         if self.auto_annotate:
#             func.__annotations__ = dict(func.__annotations__)
#             if self.on in func.__annotations__:
#                 func.__annotations__[self.on] = type[func.__annotations__[self.on]]
#             else:
#                 func.__annotations__[self.on] = list
#
#     def bind(self, func, pooled_outer):
#         if '__attach_as' is None or self.attach is False:
#             wrapped = pooled_outer
#         elif self.attach is True:
#             @wraps(func)
#             def wrapped(*args, **kwargs) -> _S:
#                 return func(*args, **kwargs)
#             wrapped.__dict__[self.attach_as] = pooled_outer
#         return wrapped
#
#     def override(self, f, okwargs, key):
#         def wrapped(*args, **kwargs):
#             def _wrapped(*args, **kwargs):
#                 pass
#             return _wrapped
#         return wrapped
#
#     def __call__(self, f):
#         self.validate_func(f)
#
#         def pooled_outer(n_cpus: Optional[int] = None, chunksize: int = 1, callback=None, error_callback=None,
#                          override_kwargs='pooled_opts'):
#             pool_kwargs = {
#                 'n_cpus': n_cpus,
#                 'chunksize': chunksize,
#                 'callback': callback,
#                 'error_callback': error_callback
#             }
#             pooled_inner = self.pooled_inner(f, pool_kwargs, override_kwargs)
#             if self.auto_annotate:
#                 self.do_auto_annotate(pooled_inner)
#             return pooled_inner
#
#         return self.bind(f, pooled_outer)


def _auto_annotate_on(on: str, func, type=List):
    func.__annotations__ = dict(func.__annotations__)
    if on in func.__annotations__:
        func.__annotations__[on] = type[func.__annotations__[on]]
    else:
        func.__annotations__[on] = list


def multiprocess(
    on: str, auto_annotate: bool = True, attach: bool = True, attach_as="pooled"
):
    """A decorator that attaches a pooled version of the function.

    .. code-block::

        import time

        @mp('y')
        def foo(x, y, z):
            time.sleep(x/10.)
            return x + y + z

        # standard call signature
        foo(1, 3, 2)

        # pooled call signature
        foo.pooled(1, range(20), 2)

    :param on: Variable to pool. The pooled version will require an iterable to be
               passed at the positional argument or key word argument indicated.
    :param auto_annotate: whether to dynamically change the annotation of the pooled function.
           This likely will not appear in the IDE.
    :param attach: if False, return only the pooled version of the function
    :param attach_as: key to attach to the original function
    :return:
    """
    if not isinstance(attach_as, str):
        raise TypeError("attach_as must be a str, not a {}".format(attach_as.__class__))
    if not valid_varname(attach_as):
        raise ValueError(
            "attach_as '{}' is not a valid variable name".format(attach_as)
        )

    def _mp(
        f: Callable[[_T], _S]
    ) -> Callable[[Callable[[_T], _S]], Callable[[_T], _S]]:

        if not isinstance(on, str):
            raise TypeError(
                "argument must be a str, found a {} {}".format(on, on.__class__)
            )
        elif not valid_varname(on):
            raise ValueError("argument '{}' is not a valid variable name".format(on))
        argspec = inspect.getfullargspec(f)
        if on not in argspec.args:
            raise ValueError(
                "argument '{}' not in signature. Use any of {}".format(on, argspec.args)
            )

        def pooled_outer(
            n_cpus: Optional[int] = None,
            chunksize: int = 1,
            callback=None,
            error_callback=None,
            override_kwargs="pooled_opts",
        ):

            outer_kwargs = {
                "n_cpus": n_cpus,
                "chunksize": chunksize,
                "callback": callback,
                "error_callback": error_callback,
            }

            @wraps(f)
            def pooled_inner(*args, **kwargs) -> List[_S]:
                # overrides
                inner_kwargs = dict(outer_kwargs)
                if override_kwargs is not None:
                    if override_kwargs in kwargs:
                        pooled_opts = kwargs[override_kwargs]
                        inner_kwargs.update(pooled_opts)
                inner_kwargs = dict(outer_kwargs)
                n_cpus = inner_kwargs["n_cpus"]
                chunksize = inner_kwargs["chunksize"]
                callback = inner_kwargs["callback"]
                error_callback = inner_kwargs["error_callback"]

                amap = argmap(f, args, kwargs)
                pooled_kwargs = []
                if not hasattr(amap[on], "__iter__"):
                    raise ValueError(
                        'argument "{}" is not iterable for pooled function'.format(on)
                    )
                for v in amap[on]:
                    _kwargs = dict(amap)
                    _kwargs[on] = v
                    pooled_kwargs.append(_kwargs)

                def starstar(f):
                    def wrapped(kwargs):
                        return f(**kwargs)

                    return wrapped

                with Pool(n_cpus) as pool:
                    run_with_pool(
                        pool,
                        f,
                        wrapper=starstar,
                        chunksize=chunksize,
                        args=pooled_kwargs,
                        callback=callback,
                        error_callback=error_callback,
                        resolved_handler=_resolved_sorted_handler,
                    )

            if auto_annotate:
                _auto_annotate_on(on, pooled_inner)
            return pooled_inner

        if attach_as is None or attach is False:
            wrapped = pooled_outer()
        elif attach is True:

            @wraps(f)
            def wrapped(*args, **kwargs) -> _S:
                return f(*args, **kwargs)

            wrapped.__dict__[attach_as] = pooled_outer

        return wrapped

    return _mp
