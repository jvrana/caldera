#################################################################################
# FIX `sys.path` and import modules
#################################################################################

import sys
import os

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.insert(0, SCRIPT_DIR)

from .utils import find_and_ins_syspath

try:
    import caldera
except ImportError:
    find_and_ins_syspath('caldera', 4)
    import caldera

#################################################################################
# MAIN
#################################################################################

