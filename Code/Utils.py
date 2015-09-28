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
import scipy as sp
import scipy.sparse as sps
import math
import cmath

def split_by_type(iter_to_split, type_fun, value_sort = None):
    """ Split a finite iterable into lists organized by element type.
    
    Convert a finite iterable into a list of lists, where all the elements of
    each list are of the same type.
    
    Parameters
    ----------
    iter_to_split : iterable
        The iterable to be split.
    type_fun : A function that returns the type of an element in a list.
    
    Returns
    -------
    out : list of lists
        All the elements in each list are of the same type.
    """
    split_dict = dict()
    for x in iter_to_split:
        type_x = type_fun(x)
        if type_x in split_dict:
            split_dict[type_x].append(x)
        else:
            split_dict[type_x] = [x]
    values = split_dict.values()
    if value_sort is not None:
        values.sort(key = value_sort)
    return values
    

def spabs(A):
    """ Utils.sparse_absolute(A)
    
    Return the elementwise absolute value of a sparse matrix.
    
    Parameters
    ----------
    A : The sparse matrix.
    
    Returns
    -------
    Aabs : The elementwise absolute value of A.
    """
    Aabs = A.copy()
    if type(Aabs) == sps.csr_matrix or type(Aabs) == sps.csc_matrix:
        Aabs.data = np.abs(Aabs.data)
    elif type(Aabs) == sps.lil_matrix:
        for i in xrange(Aabs.shape[0]):
            Aabs.data[i] = np.abs(Aabs.data[i])
    elif type(Aabs) == sps.dok_matrix:
        for k in Aabs.iterkeys():
            Aabs[k] = np.abs(Aabs[k])
    else:
        assert False
    return Aabs
    
    
def precision_lstsq(A, B, ITERATIONS = 1):
    x = np.linalg.lstsq(A, B)[0]
    for i in range(ITERATIONS):
        r = B - (A * x)
        dx = np.linalg.lstsq(A, r)[0]
        x += dx
    return x
