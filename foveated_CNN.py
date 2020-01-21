import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings

warnings.filterwarnings("error")

'''
General idea:
 - A (spiking) RNN controls the movement of fovea around image and image processing
 - A centricity like downsampling is used over the 'fovea'
'''