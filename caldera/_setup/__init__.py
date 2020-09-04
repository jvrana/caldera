from caldera._setup.setup_nx_global_access import add_global_access_to_nx
from caldera.defaults import CalderaDefaults


def setup():
    add_global_access_to_nx(CalderaDefaults.nx_global_key)
