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

import random
inputA = [random.uniform(-radius, radius) for i in range(D1*D2)]
inputB = [random.uniform(-radius, radius) for i in range(D2*D3)]

import nengo_theano as nef

net = nef.Network('Matrix Multiplication')

# make 2 matrices to store the input
net.make_array('A', N, D1*D2, radius=radius)
net.make_array('B', N, D2*D3, radius=radius)

# connect inputs to them so we can set their value
net.make_input('input A', inputA)
net.make_input('input B', inputB)
net.connect('input A','A')
net.connect('input B','B')

# the C matrix holds the intermediate product calculations
#  need to compute D1*D2*D3 products to multiply 2 matrices together
net.make_array('C', N_mult, D1*D2*D3, dimensions=2, radius=1.5*radius,
    encoders=[[1,1],[1,-1],[-1,1],[-1,-1]])

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
            
net.connect('A', 'C', transform=transformA)            
net.connect('B', 'C', transform=transformB)            
            
            
# now compute the products and do the appropriate summing
net.make_array('D', N, D1*D3, radius=radius)

def product(x):
    return x[0]*x[1]
# the mapping for this transformation is much easier, since we want to
# combine D2 pairs of elements (we sum D2 products together)    
net.connect('C', 'D', index_post=[i/D2 for i in range(D1*D2*D3)],
                      func=product)


pA = net.make_probe('A', dt_sample=0.001)
pB = net.make_probe('B', dt_sample=0.001)
pD = net.make_probe('D', dt_sample=0.001)

net.run(1)

import pylab
pylab.subplot(1,3,1)
pylab.plot(pA.get_data())
pylab.subplot(1,3,2)
pylab.plot(pB.get_data())
pylab.subplot(1,3,3)
pylab.plot(pD.get_data())
pylab.show()

