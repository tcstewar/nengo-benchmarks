"""
Nengo Benchmark Model #2: Lorenz Attractor

Input: none
Ouput: the 3 state variables for the classic Lorenz attractor
"""

N = 2000      # number of neurons
tau = 0.1     # post-synaptic time constant
sigma = 10    # Lorenz variables
beta = 8.0/3  # Lorenz variables
rho = 28      # Lorenz variables

import nengo_theano as nef
net = nef.Network('Lorenz attractor')
net.make('A', N, 3, radius=60)

def feedback(x):
    dx0 = -sigma * x[0] + sigma * x[1]
    dx1 = -x[0] * x[2] - x[1]
    dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
    return [dx0 * tau + x[0], 
            dx1 * tau + x[1], 
            dx2 * tau + x[2]]
net.connect('A', 'A', func=feedback, pstc=tau)

pState = net.make_probe('A', dt_sample=0.001)

net.run(10.0)

import pylab
pylab.plot(pState.get_data())
pylab.show()

