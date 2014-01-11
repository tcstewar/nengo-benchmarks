"""
Nengo Benchmark Model #1: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""

D = 1       # number of dimensions
L = 2       # number of layers
N = 50      # number of neurons per layer
pstc = 0.01 # synaptic time constant


import numpy as np

import nengo_theano as nef

net=nef.Network('Main', seed=1)

value = np.random.randn(D)
value /= np.linalg.norm(value)

net.make_input('input', value)

for i in range(L):
    net.make('layer%d'%i, N, D)

net.connect('input', 'layer0', pstc=pstc)
for i in range(L-1):
    net.connect('layer%d'%i, 'layer%d'%(i+1), pstc=pstc)

pInput = net.make_probe('input', dt_sample=0.001)
pOutput = net.make_probe('layer%d'%(0), dt_sample=0.001)

net.run(1.0)

import pylab
pylab.plot(pInput.get_data())
pylab.plot(pOutput.get_data())
pylab.show()

