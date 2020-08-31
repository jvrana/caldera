"""_nx_conversion.py.

Functional methods to convert networkx graphs to
:class:`caldera.data.GraphData` instances.
"""
import numpy as np

from caldera.utils import dict_join
from caldera.utils import functional as fn
from caldera.utils.tensor import to_one_hot
import functools

# TODO: expose this method

data = [{"features": True}, {"features": True}, {"features": False}]

counts_by_value = fn.compose(
    fn.group_each_by_key(lambda x: x["features"]),
    fn.index_each(0),
    fn.enumerate_each(),
    fn.map_each(lambda x: (x[1], x[0])),
    fn.return_as(dict),
)

def collate_to_one_hot(datalist, num_classes):
    print(datalist)
    print(list(fn.group_each_by_key(lambda x: x['features'])(datalist)))
    d = counts_by_value(datalist)
    print(d)
    print()
    print(datalist)
    print(list(fn.group_each_by_key(lambda x: x['features'])(datalist)))
    d = counts_by_value(datalist)
    print(d)

    to_long = fn.compose(
        fn.tee_pipe_yield(
            counts_by_value
        ),
        # fn.asterisk(
        #     lambda c, g: fn.map_each(lambda x: c[x['features']])(g)
        # ),
        list,
        # np.array,
        # functools.partial(to_one_hot, mx=num_classes)
    )
    return to_long(datalist)


oh = collate_to_one_hot(data, 10)
print(oh)