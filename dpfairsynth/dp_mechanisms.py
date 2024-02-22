from numpy.random import laplace

def Laplace_mech(x, eps, sens):
    return x + laplace(loc=0, scale=sens/eps, size=x.shape)