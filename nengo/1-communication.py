"""
Nengo Benchmark Model #1: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""


D = 1       # number of dimensions
L = 5       # number of layers
N = 50      # number of neurons per layer
pstc = 0.01 # synaptic time constant


import numpy as np
import matplotlib.pyplot as plt

import nengo

model = nengo.Model()
with model:
    value = np.random.randn(D)
    value /= np.linalg.norm(value)

    input = nengo.Node(value)

    layers = [nengo.Ensemble(N, D) for i in range(L)]

    nengo.Connection(input, layers[0])
    for i in range(L-1):
        nengo.Connection(layers[i], layers[i+1], filter=pstc)

    pInput = nengo.Probe(input, 'output')
    pOutput = nengo.Probe(layers[-1], 'decoded_output', filter=pstc)

sim = nengo.Simulator(model)
sim.run(1.0)

import pylab
pylab.plot(sim.data(pInput))
pylab.plot(sim.data(pOutput))
pylab.show()

