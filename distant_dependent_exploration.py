import spynnaker8 as p
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
import scipy.stats as spy

def weight_distribution(pop_size):
    base_weight = np.random.randn() / np.sqrt(pop_size)
    # base_weight = 0
    return base_weight

def generate_distant_dependent_from_list(sheet_dimensions, stdev):
    width = sheet_dimensions[0]
    height = sheet_dimensions[1]
    connections = []
    for width_pre in range(width):
        for height_pre in range(height):
            for width_post in range(width):
                for height_post in range(height):
                    distance = np.sqrt(np.power(width_post - width_pre, 2) + np.power(height_post - height_pre, 2))
                    pre_neuron = (height_pre * width) + width_pre
                    post_neuron = (height_post * width) + width_post
                    if np.random.random() < (spy.norm.ppf(distance / stdev) - 0.5) * 2:
                        weight = 0.1#weight_distribution(width*height)
                        connections.append([pre_neuron, post_neuron, weight, 1])

np.random.seed(272727)

p.setup(timestep=1, max_delay=250)
# input_pop = p.Population(input_size,
#                             p.SpikeSourceArray,
#                             {'spike_times': [np.linspace(0, 1000, 10) for i in range(input_size)]},
#                             label='input_pop')

# neuron = p.Population(1,
#                          p.extra_models.EPropAdaptive(**neuron_params),
#                          label='eprop_pop')

# Output population
x_dim = 10
y_dim = 10
neuron_pop_size = x_dim * y_dim
neuron_pop = p.Population(p.IF_curr_exp,
                       label="readout_pop"
                       )

input_size = neuron_pop_size
input_pop = p.Population(input_size,
                            p.SpikeSourcePoisson,
                            {'rate': 10},
                            label='input_pop')

from_list = [[i, 0, weight_distribution(input_size), i] for i in range(input_size)]
in_proj = p.Projection(input_pop,
                          neuron_pop,
                          # readout_pop,
                          p.FromListConnector(from_list),
                          p.OneToOneConnector(),
                          p.StaticSynapse(weight=0.1, delay=1),
                          label='input_connections',
                          receptor_type='excitatory')

exc_rec = p.Projection(neuron_pop,
                       neuron_pop,
                       p.DistanceDependentProbabilityConnector)

# input_pop.record('spikes')
# neuron.record('all')
# readout_pop.record('all')

runtime = 1000#cycle_time * num_repeats
p.run(runtime)
# in_spikes = input_pop.get_data('spikes')
# neuron_res = neuron.get_data('all')
# readout_res = readout_pop.get_data('all')

# Plot rec neuron output
# plt.figure()
# # plt.tight_layout()
#
# plt.subplot(4, 1, 1)
# plt.plot(neuron_res.segments[0].filter(name='v')[0].magnitude, label='Membrane potential (mV)')
#
# plt.subplot(4, 1, 2)
# plt.plot(neuron_res.segments[0].filter(name='gsyn_exc')[0].magnitude, label='gsyn_exc')
#
# plt.subplot(4, 1, 3)
# plt.plot(neuron_res.segments[0].filter(name='gsyn_inh')[0].magnitude, label='gsyn_inh')
#
# plt.subplot(4,1,4)
# plt.plot(in_spikes.segments[0].spiketrains, label='in_spikes')

plt.figure()
Figure(
    # Panel(neuron_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),
    #
    # Panel(neuron_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),
    #
    # Panel(neuron_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(in_spikes.segments[0].spiketrains, ylabel='in_spikes', xlabel='in_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

    # Panel(neuron_res.segments[0].spiketrains, ylabel='neuron_spikes', xlabel='neuron_spikes', yticks=True, xticks=True, xlim=(0, runtime)),

#     title="eprop neuron"
# )
# plt.show()
#
# plt.figure()
# Figure(
    Panel(readout_res.segments[0].filter(name='v')[0], ylabel='Membrane potential (mV)', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_exc')[0], ylabel='gsyn_exc', yticks=True, xticks=True, xlim=(0, runtime)),

    Panel(readout_res.segments[0].filter(name='gsyn_inh')[0], ylabel='gsyn_inh', yticks=True, xticks=True, xlim=(0, runtime)),

    # Panel(cycle_error, ylabel='cycle error', yticks=True, xticks=True, xlim=(0, num_repeats)),

    title="neuron data for {}".format(experiment_label)
)
# plt.show()

# Plot Readout output
# plt.figure()
# # plt.tight_layout()
#
# plt.subplot(3, 1, 1)
# plt.plot(readout_res.segments[0].filter(name='v')[0].magnitude, label='Membrane potential (mV)')
#
# plt.subplot(3, 1, 2)
# plt.plot(readout_res.segments[0].filter(name='gsyn_exc')[0].magnitude, label='gsyn_exc')
#
# plt.subplot(3, 1, 3)
# plt.plot(readout_res.segments[0].filter(name='gsyn_inh')[0].magnitude, label='gsyn_inh')


plt.show()

p.end()
print("job done")

'''
plt.figure()
plt.scatter([i for i in range(num_repeats)], )
plt.show()
'''