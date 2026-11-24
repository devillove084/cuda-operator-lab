"""Run a bundled operator driver with a reproducible seed."""
import argparse
from pathlib import Path
import random
import runpy
import sys

import numpy
import torch


parser = argparse.ArgumentParser()
parser.add_argument("driver", type=Path)
parser.add_argument("seed", type=int)
parser.add_argument("args", nargs=argparse.REMAINDER)
options = parser.parse_args()
random.seed(options.seed)
numpy.random.seed(options.seed)
torch.manual_seed(options.seed)
sys.path.insert(0, str(options.driver.resolve().parent))
sys.argv = [str(options.driver), *options.args]
runpy.run_path(str(options.driver), run_name="__main__")
