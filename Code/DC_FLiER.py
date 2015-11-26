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

from PowerNetwork import PowerNetwork
from numpy import dot
import numpy as np

def DC_FLiER(dc_mat, branches, pre_event_volts, post_event_volts, E, slack_buses, verbose=False):
    """
    FLiER using the DC approximation.
    """
    nb = dc_mat.shape[0]
    E = E[:,:nb]
    if E.shape[0] > nb:
        E = E[:nb,:]
    pre_theta = pre_event_volts[:nb]
    dvth = post_event_volts[:nb] - pre_event_volts[:nb]
    Efingerprint = E.dot(dvth)
    
    score_dict = dict()
    score_dict[-1] = np.linalg.norm(Efingerprint)
    filter_score_dict = dict()
    for branch in branches.values(): # Hack, same as not filtering.
        filter_score_dict[branch.ID] = 0.0
        
    filter_scores = sorted(filter_score_dict.items(), key = lambda x : x[1])
    min_t = np.Inf
    for branch_ind, tau in filter_scores:
        if tau > min_t:
            continue
        branch = branches[branch_ind]
        post_dc_mat = get_post_DC_mat(dc_mat, branch)
        r = dc_mat.dot(pre_theta) - post_dc_mat.dot(pre_theta)
        post_dc_mat = post_dc_mat[1:,1:]
        r = r[1:]
        try:
            fingerprint_approx = np.linalg.solve(post_dc_mat, r)
            fingerprint_approx = np.hstack([[0], fingerprint_approx])
            Efingerprint_approx = E.dot(fingerprint_approx)
            
            t = np.linalg.norm(Efingerprint_approx - Efingerprint)
            assert t > tau
            score_dict[branch_ind] = t
            if t < min_t:
                min_t = t
        except np.linalg.linalg.LinAlgError:
            score_dict[branch_ind] = np.Inf
    return score_dict, filter_score_dict, 1.0
        
def get_post_DC_subspace(powernet, branch, E):
    U = np.zeros(powernet.dc_mat.shape[0])
    bus1 = powernet.buses[branch['buses'][0]]['bus_ind']
    bus2 = powernet.buses[branch['buses'][1]]['bus_ind']
    U[bus1] =  1.0
    U[bus2] = -1.0
    U = U[1:]
    
    dc_mat = powernet.dc_mat.toarray().copy()
    dc_mat = dc_mat[1:,1:]
    sub = np.linalg.solve(dc_mat, U)
    sub = np.hstack([[0], sub])
    return (E.dot(sub))
    
    
def get_post_DC_mat(dc_mat, branch):
    dc_mat = dc_mat.toarray().copy()
    dc_approx_weight = - 1.0 / np.imag(branch.impedance)
    bus1, bus2 = [bus.index for bus in branch.buses]
    dc_mat[bus1, bus2] += dc_approx_weight
    dc_mat[bus2, bus1] += dc_approx_weight
    dc_mat[bus1, bus1] -= dc_approx_weight
    dc_mat[bus2, bus2] -= dc_approx_weight
    return dc_mat

