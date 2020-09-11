from os.path import join, isfile
import sys

def find_pkg(name: str, depth: int):
    if depth <= 0:
        ret = None
    else:
        d = ['..'] * depth
        path_parts = d + [name, '__init__.py']

        if isfile(join(*path_parts)):
            ret = d
        else:
            ret = find_pkg(name, depth - 1)
    return ret


def find_and_ins_syspath(name: str, depth: int):
    path_parts = find_pkg(name, depth)
    if path_parts is None:
        raise RuntimeError("Could not find {}. Try increasing depth.".format(name))
    path = join(*path_parts)
    if path not in sys.path:
        sys.path.insert(0, path)
