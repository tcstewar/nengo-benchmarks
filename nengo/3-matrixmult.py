"""
Nengo Benchmark Model #3: Matrix Multiplication

Input: two random matrices of size D1xD2 and D2xD3
Output: a D1xD3 matrix that is the product of the two inputs

"""

D1 = 1       # size of matrices
D2 = 2       # size of matrices
D3 = 2       # size of matrices
radius = 1   # All values must be between -radius and radius
N = 50       # number of neurons per input and output value
N_mult = 200 # number of neurons to compute a pairwise product
pstc = 0.01  # post-synaptic time constant

import random
inputA = [random.uniform(-radius, radius) for i in range(D1*D2)]
inputB = [random.uniform(-radius, radius) for i in range(D2*D3)]

import nengo
import numpy as np

model = nengo.Model()

with model:
    inA = nengo.Node(inputA)
    inB = nengo.Node(inputB)

    A = nengo.networks.EnsembleArray(nengo.LIF(D1*D2*N), D1*D2, radius=radius)
    B = nengo.networks.EnsembleArray(nengo.LIF(D2*D3*N), D2*D3, radius=radius)
    D = nengo.networks.EnsembleArray(nengo.LIF(D1*D3*N), D1*D3, radius=radius)

    encoders = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    encoders = np.tile(encoders, ((N_mult+1)/4, 1))[:N_mult]
    # the C matrix holds the intermediate product calculations
    #  need to compute D1*D2*D3 products to multiply 2 matrices together
    C = nengo.networks.EnsembleArray(nengo.LIF(D1*D2*D3*N_mult), 
            D1*D2*D3, 2,  
            radius=1.5*radius, encoders=encoders)

    nengo.Connection(inA, A.input, filter=pstc)
    nengo.Connection(inB, B.input, filter=pstc)

    #  determine the transformation matrices to get the correct pairwise
    #  products computed.  This looks a bit like black magic but if
    #  you manually try multiplying two matrices together, you can see
    #  the underlying pattern.  Basically, we need to build up D1*D2*D3
    #  pairs of numbers in C to compute the product of.  If i,j,k are the
    #  indexes into the D1*D2*D3 products, we want to compute the product
    #  of element (i,j) in A with the element (j,k) in B.  The index in
    #  A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
    #  The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
    #  two values per ensemble.  We add 1 to the B index so it goes into
    #  the second value in the ensemble.  
    transformA = [[0]*(D1*D2) for i in range(D1*D2*D3*2)]
    transformB = [[0]*(D2*D3) for i in range(D1*D2*D3*2)]
    for i in range(D1):
        for j in range(D2):
            for k in range(D3):
                transformA[(j + k*D2 + i*D2*D3)*2][j + i*D2] = 1
                transformB[(j + k*D2 + i*D2*D3)*2 + 1][k + j*D3] = 1
                
    nengo.Connection(A.output, C.input, transform=transformA, filter=pstc)            
    nengo.Connection(B.output, C.input, transform=transformB, filter=pstc)            
            
            
    # now compute the products and do the appropriate summing
    def product(x):
        return x[0]*x[1]

    C.add_output('product', product)

    import nengo.helpers
    transform = nengo.helpers.transform(D1*D2*D3, D1*D3, 
                   index_post=[i/D2 for i in range(D1*D2*D3)])
    # the mapping for this transformation is much easier, since we want to
    # combine D2 pairs of elements (we sum D2 products together)    
    nengo.Connection(C.product, D.input, transform=transform, filter=pstc)


    pA = nengo.Probe(A.output, 'output', filter=pstc)
    pB = nengo.Probe(B.output, 'output', filter=pstc)
    pD = nengo.Probe(D.output, 'output', filter=pstc)

sim = nengo.Simulator(model)
sim.run(0.5)

import pylab
pylab.subplot(1,3,1)
pylab.plot(sim.data(pA))
pylab.subplot(1,3,2)
pylab.plot(sim.data(pB))
pylab.subplot(1,3,3)
pylab.plot(sim.data(pD))
pylab.show()


