from caldera.utils import functional as Fn


def feature_info(g, global_key: str = None):
    collect_values_by_key = Fn.compose(
        Fn.map_each(lambda x: x.items()),
        Fn.chain_each(),
        Fn.group_each_by_key(lambda x: x[0]),
        Fn.map_each(lambda x: (x[0], [_x[1] for _x in x[1]])),
        dict,
    )  # from a list of dictionaries, List[Dict] -> Dict[str, List]

    unique_value_types = Fn.compose(
        Fn.index_each(-1),
        collect_values_by_key,
        lambda x: {k: {_v.__class__ for _v in v} for k, v in x.items()},
    )

    unique_value_types(g.nodes(data=True))

    return {
        "node": {"keys": unique_value_types(g.nodes(data=True))},
        "edge": {"keys": unique_value_types(g.edges(data=True))},
        "global": {"keys": unique_value_types([(g.get_global(global_key),)])},
    }
