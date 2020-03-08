from ctypes import *
import numpy as np

# Load necessary library we compiled
_cuda_lib = CDLL('./libcompute.so')
