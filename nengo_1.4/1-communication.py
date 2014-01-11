"""
Nengo Benchmark Model #1: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""

D = 1       # number of dimensions
L = 2       # number of layers
N = 50      # number of neurons per layer
pstc = 0.01 # synaptic time constant

import nef
import numeric as np

net = nef.Network('Benchmark-1 Communication', seed=1)

import random
value = np.array([random.gauss(0,1) for i in range(D)])
value /= np.norm(value)

net.make_input('input', value)

for i in range(L):
    net.make('layer%d'%i, N, D)

net.connect('input', 'layer0', pstc=pstc)
for i in range(L-1):
    net.connect('layer%d'%i, 'layer%d'%(i+1), pstc=pstc)

net.view()

