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
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import time

def FLiER(Y, branches, pre_event_volts, post_event_volts, E, buses, slack_buses,
          use_filter=True, verbose=False):
    """ FLiER

    Runs FLiER, as described in the paper.

    Inputs:
    Y - The admittance matrix of the network.
    branches - A list of branches in the network.
    pre_event_volts - The steady-state voltages in the network prior to line 
                      failure. Voltages in polar form.
    post_event_volts - The observed steady-state voltages in the network after
                       line failure. Has the same format as pre_event_volts,
                       but only those indices rendered observable by PMUs
                       are valid.
    E - The matrix that projects voltage vectors onto the subspace rendered
        observable by the PMUs.
    bus_type - A list that state the type of bus at each bus in the network.
    slack_buses - A list of indices of slack buses.
    use_filter - Set to False to bypass the filter and compute tij for every
                 line.
                 
    Outputs:
    score_dict - A dictionary with keys that are indices into the 'branches' list
                 and values that are tij's. The "no line failure" case has
                 key -1.
    filter_score_dict - Same format as score_dict, but contains filter scores.
    ts_computed - The number of tij's computed.

    """
    nb = Y.shape[0]    
    
    # Compute indices of polar voltage components not specified by
    # network parameters.
    unspecified_vars = np.hstack([np.ones(nb), np.array([bus.type for bus in buses]) == 'PQ'])
    for i in slack_buses:
        unspecified_vars[i] = 0
        unspecified_vars[nb+i] = 0
    unspecified_inds = np.nonzero(unspecified_vars)[0]
    
    ind2redind = np.zeros(2*nb)
    ind = 0
    for i in xrange(2*nb):
        if unspecified_vars[i]:
            ind2redind[i] = ind
            ind += 1
        else:
            ind2redind[i] = -1
    
    # Compute the reduced projection matrix and the observed voltage
    # component changes associated with unspecified indices.
    Ered = E[:,unspecified_inds]
    
    # Compute the fingerprint.
    dvm = post_event_volts[nb:] - pre_event_volts[nb:]
    dvth = post_event_volts[:nb] - pre_event_volts[:nb]
    tilde_fingerprint = np.hstack([pre_event_volts[nb:] * dvth, dvm])
    fingerprint = tilde_fingerprint[unspecified_inds]
    Efingerprint = dot(Ered, fingerprint)

    # Compute J, the reduced Jacobian at the pre-event voltages.
    J = spowerflow_jacob(pre_event_volts, Y, ind2redind, len(unspecified_inds))
    JT = sps.csc_matrix(J.T)
    JTfac = spsl.factorized(JT)
    
    powerflowpreVY = powerflow(pre_event_volts, Y)
    EJinv = sps.csc_matrix(JTfac(Ered.T).T).todense(order='F')
    
    start_time = time.clock()
    infodict = compute_line_info(Y, branches, unspecified_inds, pre_event_volts,
                                 EJinv, ind2redind, powerflowpreVY)
    # Compute the tau filter scores.
    
    filter_score_dict = get_filter_scores(branches, infodict, Efingerprint, Ered, J)
    filter_scores = sorted(filter_score_dict.iteritems(), key= lambda x: x[1])
    mid_time = time.clock()
    # Compute the tij scores.
    out = get_fingerprint_scores(branches, EJinv, JTfac, infodict, filter_scores,
                                 Efingerprint, use_filter = use_filter)
    score_dict, fraction_ts_computed = out
    end_time = time.clock()
    if verbose:
        print_str = "Filtering time: {0}, Fingerprint time: {1}, Total time: {2}"
        print print_str.format(mid_time - start_time, end_time - mid_time, 
                               end_time - start_time)
        
    return score_dict, filter_score_dict, fraction_ts_computed
    
def compute_line_info(Y, branches, unspecified_inds, pre_event_volts, EJinv, 
                      ind2redind, powerflowpreVY):
    infodict = dict()
    for branch_ind in range(len(branches)):
        branch = branches[branch_ind]
        
        r = get_r(pre_event_volts, Y, branch, unspecified_inds, powerflowpreVY)
        EJinvr = -EJinv.dot(r)
        
        Wbar, Vbar = get_Wbar_Vbar(pre_event_volts, branch, unspecified_inds, 
                                   ind2redind)
      
        subspace = EJinv * Wbar
        infodict[branch_ind] = (r, EJinvr, Wbar.toarray(), Vbar.toarray(), 
                                subspace)
    return infodict
    
    
def get_filter_scores(branches, infodict, Efingerprint, Ered, J):
    filter_score_dict = dict()
    for branch_ind in range(len(branches)):
        r, EJinvr, Wbar, Vbar, subspace = infodict[branch_ind]
        x = np.linalg.lstsq(subspace, Efingerprint)[0]
        filter_score_dict[branch_ind] = np.linalg.norm(Efingerprint - subspace.dot(x))
    return filter_score_dict


def get_fingerprint_scores(branches, EJinv, JTfac, infodict, filter_scores, 
                           Efingerprint, use_filter = True):
    score_dict = dict()
    score_dict[-1] = np.linalg.norm(Efingerprint)
    
    ts_computed = 0
    M = np.inf
    for (i, s_ij) in filter_scores:
        if use_filter and M < s_ij:
            continue
        branch = branches[i]
        
        r, EJinvr, Wbar, Vbar, subspace = infodict[i]
        VbarTJinv = JTfac(Vbar).T
        Xik = VbarTJinv.dot(Wbar)
        smw_mid = np.eye(3) + Xik
        try:
            z = np.linalg.solve(smw_mid, VbarTJinv.dot(-r))
        except np.linalg.linalg.LinAlgError:
            score_dict[i] = np.Inf
        Edv = EJinv.dot(-r) - EJinv.dot(Wbar.dot(z))
        tij = np.linalg.norm(Edv - Efingerprint)
        
        score_dict[i] = tij
        
        if tij < M:
            M = tij
        ts_computed += 1
    
    frac_comp = (float(ts_computed)+1) / (len(filter_scores) + 1)
    return score_dict, frac_comp

def get_Wbar_Vbar(pre_event_volts, branch, unspecified_inds, ind2redind):
    """
    Compute Wbar and Vbar as described in Equation (17) of the paper.
    """
    nb = len(pre_event_volts) / 2
    W, V = compute_WV(pre_event_volts, branch)
    U = get_U(nb, branch, unspecified_inds, ind2redind)
    Wbar = U.dot(sps.csr_matrix(W))
    Vbar = U.dot(sps.csr_matrix(V))
    return Wbar.tocsc(), Vbar

def compute_WV(oldv_polar, branch):
    """
    Compute W and V as described in Equation (12) of the paper.
    """
    nb = len(oldv_polar) / 2
    i, k = [bus.index for bus in branch.buses]
    
    # Compute relevant admittance quantities associated with \tilde A, 
    # as in Equation (10) of the paper.
    
    # branch_imp = branch['resistance'] + 1j * branch['reactance']
    # branch_adm = 1.0 / branch_imp
    branch_adm = branch.admittance
    if not branch.nonunity_tap:
        yik = -branch_adm
        yki = -branch_adm
        yii = branch_adm + 1j*branch.line_charging_susc / 2.0
        ykk = branch_adm + 1j*branch.line_charging_susc / 2.0
    else:
        tap = branch.tap
        yii = (branch_adm + 1j * branch.line_charging_susc / 2.0) / (tap*np.conj(tap))
        ykk = (branch_adm + 1j * branch.line_charging_susc / 2.0)
        yik = -branch_adm / np.conj(tap)
        yki = -branch_adm / tap

    vMi = oldv_polar[nb+i]
    vMk = oldv_polar[nb+k]
    thetai = oldv_polar[i]
    thetak = oldv_polar[k]
    thetaik = thetai - thetak
                
    gii, gkk = np.real((yii, ykk))
    bii, bkk = np.imag((yii, ykk))
    gikcos, gkicos = np.real((yik, yki)) * np.cos(thetaik)
    bikcos, bkicos = np.imag((yik, yki)) * np.cos(thetaik)
    giksin, gkisin = np.real((yik, yki)) * np.sin(thetaik)
    biksin, bkisin = np.imag((yik, yki)) * np.sin(thetaik)

    Pik = vMi*vMk*(gikcos+biksin) + gii*vMi*vMi
    Pki = vMi*vMk*(gkicos-bkisin) + gkk*vMk*vMk    
    Qik = vMi*vMk*(-bikcos+giksin) - bii*vMi*vMi
    Qki = vMi*vMk*(-bkicos-gkisin) - bkk*vMk*vMk
    
    L = np.array([[ Qik + bii*vMi**2, -Pik - gii*vMi**2, -Pik + gii*vMi**2],
                  [-Qki - bkk*vMk**2, -Pki + gkk*vMk**2, -Pki - gkk*vMk**2],
                  [-Pik + gii*vMi**2, -Qik + bii*vMi**2, -Qik - bii*vMi**2],
                  [ Pki - gkk*vMk**2, -Qki - bkk*vMk**2, -Qki + bkk*vMk**2]])
    D = np.diag([1.0, 1.0/vMi, 1.0/vMk])
    W = np.dot(L,D)
        
    V = np.array([[ 1, 0, 0],
                  [-1, 0, 0],
                  [ 0, 1, 0],
                  [ 0, 0, 1]])
    
    return W, V
    
def get_r(pre_event_volts, Y, branch, unspecified_inds, powerflowpreVY):
    """
    Compute r (reduced version) as described in Equations __ and (8) of the paper.
    """
    postY = Y.copy()
    PowerNetwork.remove_line_from_Y(postY, branch)
    r = powerflow(pre_event_volts, postY) - powerflowpreVY
    r = r[unspecified_inds]
    return r

def get_U(nb, branch, unspecified_inds, ind2redind):
    """
    Compute U (reduced version) as described in Equation (16) of the paper.
    """
    i, k = [bus.index for bus in branch.buses]
    line_inds = [i, k, nb+i, nb+k]
    rows_to_keep = [line_inds[x] for x in range(4) if line_inds[x] 
                    in unspecified_inds]
    cols_to_keep = [x for x in range(4) if line_inds[x] in unspecified_inds]
    
    ij = np.zeros((2, 0))
    data = []
    for i in range(len(rows_to_keep)):
        if ind2redind[rows_to_keep[i]] != -1:
            rowind = ind2redind[rows_to_keep[i]]
            colind = cols_to_keep[i]
            ij = np.hstack([ij, np.array([[rowind], [colind]])])
            data.append(1)
    U = sps.csr_matrix((data, ij), shape=(len(unspecified_inds), 4))
            
    # U2 = np.zeros((2*nb, 4))
    # U2[rows_to_keep, cols_to_keep] = 1
    # U2 = U2[unspecified_inds,:]
    # assert np.linalg.norm(U2 - U.toarray()) < 1e-12
    return U

def powerflow(x, Y):
    """
    Compute the power output of the power flow equations.
    
    Inputs:
    x - Network bus voltages. First nb entries are voltage phase angles, next
       nb entries are voltage magnitudes.
    Y - Network complex admittance matrix.
    
    Output:
    Power injections at each bus. First nb entries of the output are real
    power injections, next nb entries are reactive power injections.
    """
    nb = Y.shape[0]
    out = np.zeros(2*nb)
    vC = x[nb:] * np.exp(1j * x[:nb])
    s = np.multiply(vC, np.conj(Y.dot(vC)))
    out[:nb] = np.real(s)
    out[nb:] = np.imag(s)
    return out

# def powerflow_jacob(x, Y):
#     """
#     Compute the Jacobian of the power flow equations.
    
#     Inputs:
#     x - Network bus voltages. First nb entries are voltage phase angles, next
#        nb entries are voltage magnitudes.
#     Y - Network complex admittance matrix.
    
#     Output:
#     The Jacobian of the power flow equations. The i'th row of the Jacobian is 
#     the gradient of the i'th element of the output of the power flow equations.
#     """
#     nb = Y.shape[0]
#     vM = x[nb:]
#     vTheta = x[:nb]
#     vC = vM * np.exp(1j * vTheta)
#     vCT = vC[:, np.newaxis]
#     s = powerflow(x, Y)
#     s = s[:nb] + 1j * s[nb:]
    
#     J11 = np.imag(vCT * np.conj(Y * vC)) - np.imag(np.diag(s))
#     J12 = np.real(vCT * np.conj(Y * np.exp(1j * vTheta)) +
#                   np.diag(np.exp(1j * vTheta) * np.conj(np.dot(Y, vC))))
#     J21 = -np.real(vCT * np.conj(Y * vC)) + np.real(np.diag(s))
#     J22 = np.imag(vCT * np.conj(Y * np.exp(1j * vTheta)) +
#                   np.diag(np.exp(1j * vTheta) * np.conj(np.dot(Y, vC))))
    
#     return np.vstack([np.hstack([J11, J12]),
#                       np.hstack([J21, J22])])

                      
def spowerflow_jacob(x, Y, ind2redind, rnb):
    def add_element(i, j, val):
        if ind2redind[i] != -1 and ind2redind[j] != -1:
            ij.append(ind2redind[i])
            ij.append(ind2redind[j])
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
    J = sps.csc_matrix((data, ij), shape=(rnb, rnb))

    return J
