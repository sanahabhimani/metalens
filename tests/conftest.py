# make fixtures for:
# fake_cutparams_file
# fake spindle cal file
# fake metrology file plane
# fake config yaml files (3 -- spindles.yaml, lenstesttouches.yaml, lensparams.yaml)

import sys
import os

# Ensure the metalens root is importable from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
