from __future__ import annotations
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt; plt.style.use('seaborn-pastel'); plt.set_cmap('bone')
import matplotlib.image as mpimg
import neurite as ne
import numpy as np
import os; os.environ['VXM_BACKEND'] = 'pytorch'
import sys
from time import perf_counter
import torch
import torchvision
from tqdm import tqdm 
import random
import voxelmorph as vm
print(sys.version)
print('numpy version', np.__version__)
print('torch version', torch.__version__)
print(f'voxelmorph using {vm.py.utils.get_backend()} backend')