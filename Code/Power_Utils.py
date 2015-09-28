"""This file is part of the FLiER Test Suite.

    The FLiER Test Suite is free software: you can redistribute it 
	and/or modify it under the terms of the GNU General Public License as 
	published by the Free Software Foundation, either version 3 of the 
	License, or (at your option) any later version.

    The FLiER Test Suite is distributed in the hope that it will be 
	useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with the FLiER Test Suite.  If not, see 
	<http://www.gnu.org/licenses/>.
	
	Copyright Colin Ponce 2015.
"""

import numpy as np
import scipy.sparse as sps

def powerflow(x, Y):
    if x.dtype == np.complex:
        return x * np.conj(Y.dot(x))
    else:
        nb = Y.shape[0]
        out = np.zeros(2*nb)
        vC = x[nb:] * np.exp(1j * x[:nb])
        s = np.multiply(vC, np.conj(Y.dot(vC)))
        out[:nb] = np.real(s)
        out[nb:] = np.imag(s)
    return out
    
def powerflow_jacob(x, Y):
    def add_element(i, j, val):
        ij.append(i)
        ij.append(j)
        data.append(val)
    
    nb = Y.shape[0]
    vM = x[nb:]
    vTheta = x[:nb]
    vC = vM * np.exp(1j * vTheta)
    vA = np.exp(1j*vTheta)
    s = powerflow(x, Y)
    s = s[:nb] + 1j * s[nb:]
    
    ij = []
    data = []

    for i,j in zip(*Y.nonzero()):
        add_element(   i,    j, np.imag( vC[i] * np.conj(Y[i,j] * vC[j])))
        add_element(   i, nb+j, np.real( vC[i] * np.conj(Y[i,j] * vA[j])))
        add_element(nb+i,    j, np.real(-vC[i] * np.conj(Y[i,j] * vC[j])))
        add_element(nb+i, nb+j, np.imag( vC[i] * np.conj(Y[i,j] * vA[j])))
    YvC = Y.dot(vC)
    for i in xrange(nb):
        if s[i] == 0 and YvC[i] == 0:
            continue
        add_element(   i,    i, -np.imag(s[i]))
        add_element(   i, nb+i,  np.real(vA[i] * np.conj(YvC[i])))
        add_element(nb+i,    i,  np.real(s[i]))
        add_element(nb+i, nb+i,  np.imag(vA[i] * np.conj(YvC[i])))
        
    ij = np.reshape(ij, (2, len(ij)/2), order='F')
    return sps.csr_matrix((data, ij), shape=(2*nb, 2*nb))
