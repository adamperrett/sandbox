import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings

warnings.filterwarnings("error")

'''
General idea:
 - neurons with varying plasticity rules compete 
 - some form of local or global resource is used to determine fitness
 - if a neuron contributes something useful to learning give resource, else don't
 - possibly compete at synapse level
 - possibly its the combination of pre and post neuron that creates the learning rule
 - reward signal controls resource acquiring or it's a local rule related to firing
 - synapses 'dying', maybe related to post neurons dying, effects fitness
 - global or local neuromodulators affect plasticity rules
'''