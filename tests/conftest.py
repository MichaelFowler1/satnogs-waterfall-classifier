# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import sys

# Tests import the pipeline modules (dataset.py, train.py, infer.py) from the
# repo root, one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
