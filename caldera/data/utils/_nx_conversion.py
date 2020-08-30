"""
_nx_conversion.py

Functional methods to convert networkx graphs to :class:`caldera.data.GraphData` instances.
"""

from caldera.utils import dict_join
import numpy as np

# TODO: expose this method
from caldera.utils.tensor import to_one_hot

data = [
    {'features': True},
    {'features': True},
    {'features': False}
]

from caldera.utils import fn

f = fn.pipe(
    fn.group_each_by_key(lambda x: x['features']),
    fn.index_each(0),
    fn.enumerate_each(),
    fn.map_each(lambda x: (x[1], x[0])),
    fn.return_as(dict)
)

print(f(data))