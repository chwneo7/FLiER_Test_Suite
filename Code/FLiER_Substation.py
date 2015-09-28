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
from Bus import Ring_Substation
from numpy import dot
from Power_Utils import powerflow, powerflow_jacob
from Utils import precision_lstsq
import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
import matplotlib.pyplot as plt
import time

class FLiER_Substation(object):

    def __init__(self, powernet, pre_event_volts, pmus, 
                 eager_construction = True, verbose=False):
        self.v  = pre_event_volts
        self.powernet         = powernet
        self.n                = powernet.nb
        self.Y                = powernet.Y
        self.verbose = verbose
        self.pmus = pmus
        self._en = None
        self._num_slave_nodes = None
        self._substations = None
        self._rev = None
        self._ev = None
        self._eindr2reindr = None
        self._reindr2eindr = None
        self._extY = None
        self._dHdv = None
        self._C1_columns = None
        self._C2_columns = None
        self._C = None
        self._A = None
        self._Afac = None
        self._E = None
        self._EAinv = None
        self._eb = None
        self._lamed = None
        
        if eager_construction:
            x = self.substations
            x = self.num_slave_nodes
            x = self.rev
            x = self.ev
            x = self.eindr2reindr
            x = self.reindr2eindr
            x = self.extY
            x = self.dHdv
            x = self.C
            x = self.A
            x = self.Afac
            x = self.E
            x = self.EAinv
            x = self.eb
            x = self.lamed
                
    @property
    def substations(self):
        """A list of Ring_Substation objects in the network."""
        if self._substations is None:
            substations = [0]*self.n
            # Create the set of Ring_Substations
            for bus in self.powernet.buses:
                substations[bus.index] = Ring_Substation(bus)
            assert not any([sub == 0 for sub in substations])
            assert len(substations) == self.n
            # Connect Ring_Substations together.
            for branch in self.powernet.branches.values():
                sub1 = substations[branch.buses[0].index]
                sub2 = substations[branch.buses[1].index]
                Ring_Substation.connect_nodes(sub1, sub2, branch)
            # Set extended indices.
            eind = 0
            for sub in substations:
                sub.set_extended_indices(eind)
                eind += sub.num_nodes()
            self._en = eind
            self._substations = substations
            
        return self._substations
        
    @property
    def en(self):
        """The 'extended' size of the network, i.e. the number of 
        substation nodes."""
        assert self._en is not None # created in self.substations()
        return self._en
        
    @property
    def num_slave_nodes(self):
        """The number of slave nodes in this network."""
        if self._num_slave_nodes is None:
            self._num_slave_nodes = self.en - self.n
        return self._num_slave_nodes
        
    def branch_endpoint_einds(self, branch):
        """Get the extended indices of the nodes at either 
        end of this branch."""
        bus0ind = branch.buses[0].index
        node0ind = self.substations[bus0ind].branch2node(branch).eind
        bus1ind = branch.buses[1].index
        node1ind = self.substations[bus1ind].branch2node(branch).eind
        return node0ind, node1ind
                
    @property
    def eind2node(self):
        """A list that converts extended index into node."""
        if self._eind2node is None:
            self._eind2node = [0]*self.en
            for sub in self.substations:
                for node in sub.nodes:
                    self._eind2node[node.eind] = node
        return self._eind2node
        
    def set_reordering_lists(self):
        """See the doc for eind2reindr."""
        self._eindr2reindr = np.zeros(2*self.en, dtype=int)
        self._reindr2eindr = np.zeros(2*self.en, dtype=int)
        curr_master_ind = 0
        curr_slave_ind = 2*self.n
        enmn = self.en - self.n
        for sub in self.substations:
            master_eind = sub.nodes[0].eind
            self._eindr2reindr[master_eind] = curr_master_ind
            self._eindr2reindr[master_eind+self.en] = curr_master_ind+self.n
            self._reindr2eindr[curr_master_ind] = master_eind
            self._reindr2eindr[curr_master_ind+self.n] = master_eind+self.en
            curr_master_ind += 1
            for node in sub.nodes[1:]:
                eind = node.eind
                self._eindr2reindr[eind] = curr_slave_ind
                self._eindr2reindr[eind + self.en] = curr_slave_ind + enmn
                self._reindr2eindr[curr_slave_ind] = eind
                self._reindr2eindr[curr_slave_ind + enmn] = eind + self.en
                curr_slave_ind += 1

    @property
    def eindr2reindr(self):
        """eind is extended index, with length en. 
        eindr is extended index for real formulation, which has length 2*en,
        and maps every node to a pair of indices, for magnitude and phase 
        angle. 
        reind is reordered extended index, which puts master nodes first, 
        followed by slave nodes.
        reindr is reordered extended index for real formulation, which has order
        (master node magnitude variables), (master node phase angle variables),
        (slave node magnitude variables), (slave node phase angle variables)"""
        if self._eindr2reindr is None:
            self.set_reordering_lists()
        return self._eindr2reindr
        
    @property
    def reindr2eindr(self):
        """See the doc for eind2reindr. This is the inverse."""
        if self._reindr2eindr is None:
            self.set_reordering_lists()
        return self._reindr2eindr
        
    @property
    def extY(self):
        """The extended node-wise admittance matrix. Has size en x en rather
        than n x n."""
        if self._extY is None:
            extY = sps.lil_matrix((self.en, self.en), dtype=np.complex)
            for branch in self.powernet.branches.values():
                node0ind, node1ind = self.branch_endpoint_einds(branch)
                edge_adm = branch.admittance
                Yii = (edge_adm + 1j*branch.line_charging_susc / 2.0)
                if branch.nonunity_tap:
                    tap = branch.tap
                    extY[node0ind, node0ind] += Yii / (tap*np.conj(tap))
                    extY[node1ind, node1ind] += Yii
                    extY[node0ind, node1ind] -= edge_adm / np.conj(tap)
                    extY[node1ind, node0ind] -= edge_adm / tap
                else:
                    extY[node0ind, node0ind] += Yii
                    extY[node1ind, node1ind] += Yii
                    extY[node0ind, node1ind] -= edge_adm
                    extY[node1ind, node0ind] -= edge_adm
                    
            for sub in self.substations:
                for node in sub.nodes:
                    extY[node.eind, node.eind] += node.shunt_adm 
            self._extY = extY
        return self._extY
        
    @property
    def E(self):
        """A matrix that project a voltage vector onto the observable subspace.

        The observable subspace is axis-aligned and associated with voltage
        magnitudes and phase angles associatd with those buses that the PMUs
        can observe (the PMU buses and their immediate neighbors).
        """
        if self._E is None:
            phase_vars = []
            mag_vars = []
            buses_observed = []
            for p in self.pmus:
                sub = self.substations[p]
                if p not in buses_observed:
                    master_eind = sub.nodes[0].eind
                    phase_vars.append(self.eindr2reindr[master_eind])
                    mag_vars.append(self.eindr2reindr[master_eind+self.en])
                    buses_observed.append(p)
                for node in sub.nodes[2:]:
                    opp_sub_ind = node.opposing_node.substation.index
                    if opp_sub_ind not in buses_observed:
                        opp_node_eind = node.opposing_node.eind
                        phase_vars.append(self.eindr2reindr[opp_node_eind])
                        mag_vars.append(self.eindr2reindr[opp_node_eind+self.en])
                        buses_observed.append(opp_sub_ind)
            lpn = len(phase_vars) + len(mag_vars)
            assert lpn == 2 * len(phase_vars)
            ijinds = np.vstack([[np.arange(lpn)],
                                [np.hstack([phase_vars, mag_vars])]])
            shape = (lpn, self.dHdv.shape[0] + self.C.shape[1])
            self._E = sps.csr_matrix((np.ones(lpn), ijinds), shape=shape)
        return self._E
        
    @property
    def eb(self):
        """Power injections, extended, by node."""
        if self._eb is None:
            eb = np.zeros(2*self.en)
            for sub in self.substations:
                for node in sub.nodes:
                    injection = node.generation - node.load
                    eb[node.eind] = np.real(injection)
                    eb[node.eind+self.en] = np.imag(injection)
            self._eb = eb
        return self._eb
        
    @property
    def lamed(self):
        """lamed, the Phoenician name for lambda. (As the word lambda is 
        reserved in Python.)"""
        if self._lamed is None:
            eH = powerflow(self.ev, self.extY)
            resid = self.eb - eH
            rresid = resid[self.reindr2eindr]
            
            out = spsl.lsqr(self.C, rresid, atol=1e-14, btol=1e-14)
            # assert out[3] < 1e-8
            self._lamed = np.matrix(out[0]).T
        return self._lamed
        
    @staticmethod
    def extend_vector(substations, vec, en):
        """Convert a bus-level voltage set into a node-level voltage set."""
        n = len(vec) / 2
        ext_vec = np.zeros(2*en, dtype=vec.dtype)
        assert en == substations[-1].nodes[-1].eind+1
        for sub in substations:
            for node in sub.nodes:
                ext_vec[node.eind] = vec[sub.index]
                ext_vec[node.eind+en] = vec[sub.index+n]
        return ext_vec
        
    @staticmethod
    def extend_complex_vector(substations, vec, en):
        n = len(vec)
        ext_vec = np.zeros(en, dtype=complex)
        assert en == substations[-1].nodes[-1].eind+1
        for sub in substations:
            for node in sub.nodes:
                ext_vec[node.eind] = vec[sub.index]
        return ext_vec

    @property
    def ev(self):
        """Extended voltages. The voltage of each substation node in the 
        network, as opposed to each substation (or bus) in the network."""
        if self._ev is None:
            self._ev = FLiER_Substation.extend_vector(self.substations,
                                                      self.v, self.en)
        return self._ev
        
    @property 
    def rev(self):
        """Reordered extended voltages."""
        if self._rev is None:
            self._rev = self.ev[self.reindr2eindr]
        return self._rev
                
    @property
    def dHdv(self):
        """Compute the reordered extended Jacobian matrix."""
        if self._dHdv is None:
            en = self.en
            extY = self.extY
            e2r = self.eindr2reindr
            vM = self.ev[en:]
            vTheta = self.ev[:en]
            vC = vM * np.exp(1j * vTheta)
            vA = np.exp(1j*vTheta)
            s = powerflow(self.ev, extY)
            s = s[:en] + 1j * s[en:]
            ij = []
            data = []
            def add_element(i,j,val):
                ij.append(i)
                ij.append(j)
                data.append(val)
            for i,j in zip(*extY.nonzero()):
                val = np.imag(vC[i] * np.conj(extY[i,j] * vC[j]))
                add_element(e2r[i], e2r[j], val)
                
                val = np.real(vC[i] * np.conj(extY[i,j] * vA[j]))
                add_element(e2r[i], e2r[en+j], val)
                
                val = np.real(-vC[i] * np.conj(extY[i,j] * vC[j]))
                add_element(e2r[en+i], e2r[j], val)
                
                val = np.imag(vC[i] * np.conj(extY[i,j] * vA[j]))
                add_element(e2r[en+i], e2r[en+j], val)
                
            extYvC = extY.dot(vC)
            for i in xrange(en):
                if s[i] == 0 and extYvC[i] == 0:
                    continue
                add_element(e2r[   i], e2r[   i], -np.imag(s[i]))
                add_element(e2r[   i], e2r[en+i],  
                            np.real(vA[i] * np.conj(extYvC[i])))
                add_element(e2r[en+i], e2r[   i],  np.real(s[i]))
                add_element(e2r[en+i], e2r[en+i],  
                            np.imag(vA[i] * np.conj(extYvC[i])))
                
            ij = np.reshape(ij, (2, len(ij)/2), order='F')
            self._dHdv = sps.csr_matrix((data, ij), shape=(2*en, 2*en))
        return self._dHdv
        
    def get_C1_columns(self, bus_ind):
        """
        The matrix C1 is as described in the comment for C. Every PV
        and Slack node in the network is given a column in C1. This
        function maps PV and slack nodes to columns of C1.
        """
        if self._C1_columns is None:
            self._C1_columns = [0]*self.n
            curr_col = 0
            for sub in self.substations:
                if sub.type == 'PQ':
                    self._C1_columns[sub.index] = None
                elif sub.type == 'PV':
                    self._C1_columns[sub.index] = (curr_col,)
                    curr_col += 1
                else:
                    assert sub.type == 'Slack'
                    self._C1_columns[sub.index] = (curr_col, curr_col+1)
                    curr_col += 2
        return self._C1_columns[bus_ind]
        
    def get_C2_columns(self, node_eind):
        """
        The matrix C2 is as described in the comment for C. Every slave
        node is given a column in C2. This function maps slave node
        indices to columns of C2. This is also used to index into the
        rows of the matrix U.
        """
        if self._C2_columns is None:
            self._C2_columns = [0]*self.en
            curr_col = 0
            for sub in self.substations:
                self._C2_columns[sub.nodes[0].eind] = None
                for node in sub.nodes[1:]:
                    self._C2_columns[node.eind] = (curr_col, curr_col+1)
                    curr_col += 2
        return self._C2_columns[node_eind]
        
    @property
    def C(self):
        """This matrix has two parts. 
        C1: Allows PV and Slack inds to add arbitrary amounts of reactive 
            power (PV) or real and reactive powers (Slack) to their power 
            injections.
        C2: Allows a substation to move power injection freely around
            the various nodes in the substation."""
        if self._C is None:
            rows = self.dHdv.shape[0]
            num_C1_cols = len(self.powernet.pv_inds) + 2 * len(self.powernet.slack_inds)
            num_C2_cols = 2 * self.num_slave_nodes
            C1 = sps.lil_matrix((rows, num_C1_cols))
            for sub in self.substations:
                c1_cols = self.get_C1_columns(sub.index)
                if sub.type == 'PQ':
                    assert c1_cols is None
                    continue
                elif sub.type == 'PV':
                    assert len(c1_cols) == 1
                    C1[self.eindr2reindr[sub.nodes[0].eind+self.en], c1_cols[0]] = 1
                else:
                    assert sub.type == 'Slack'
                    assert len(c1_cols) == 2
                    C1[self.eindr2reindr[sub.nodes[0].eind], c1_cols[0]] = 1
                    C1[self.eindr2reindr[sub.nodes[0].eind+self.en], c1_cols[1]] = 1
                    
            C2 = sps.lil_matrix((rows, num_C2_cols))
            for sub in self.substations:
                master_mag_ind = self.eindr2reindr[sub.nodes[0].eind]
                master_phase_ind = self.eindr2reindr[sub.nodes[0].eind+self.en]
                assert master_mag_ind == sub.index
                assert master_phase_ind == sub.index + self.n
                for node in sub.nodes[1:]:
                    curr_mag_ind = self.eindr2reindr[node.eind]
                    curr_phase_ind = self.eindr2reindr[node.eind+self.en]
                    c2_cols = self.get_C2_columns(node.eind)
                    assert c2_cols is not None
                    C2[curr_phase_ind  , c2_cols[0]] =  1
                    C2[master_phase_ind, c2_cols[0]] = -1
                    C2[curr_mag_ind    , c2_cols[1]] =  1
                    C2[master_mag_ind  , c2_cols[1]] = -1
            C = sps.bmat([[C1, C2]], format='csr')
            self._C = C
        return self._C
        
    @property
    def A(self):
        """ The matrix A as described on Page 3 of the paper."""
        if self._A is None:
            self._A = sps.bmat([[self.dHdv,  self.C],
                               [self.C.T, None]], format='csc').real
        return self._A
        
    @property
    def Afac(self):
        """ A pre-factorized A matrix to be used in linear solves."""
        if self._Afac is None:
            self._Afac = spsl.factorized(self.A)
        return self._Afac
        
    @property
    def EAinv(self):
        """ $E A^{-1}$, the projection of $A^{-1}$ onto the observable 
        subspace."""
        if self._EAinv is None:
            AT = self.A.T.tocsc()
            self._EAinv = (spsl.spsolve(AT, self.E.T).T).todense(order='F')
        return self._EAinv

        
        

###Enter FLiER Code
        
    def find_topology_error(self, post_event_volts, test_type, use_filter=True):
        """ Run FLiER for substation reconfigurations.

        *Contingencies have the following format: A tuple.
           Element 0: The index of the substation that split.
           Element 1: The set of nodes of that substation that split away.
        Note that single line failures can be described as the event (sub, node),
        where sub is one of the substations attached to the failed line and node
        is the node attached to the line that failed.

        Inputs:
        post_event_volts (List of doubles) - A list of the observable voltages.
           Contains phase angles followed by voltage magnitudes.
        test_type ("Full", "Single_Lines") - Whether this is a full contingency
           search or a line failure search.
        use_filter (True, False) - Whether or not to use FLiER's filter. Default
           is True.

        Output:
        score_dict (dictionary) - A dictionary with keys that are contingencies*
           and values that are t_ij scores. The "no contingency" case has key -1.
        filter_score_dict (dictionary) - Same format as score_dict, but contains
           filter scores tau_ij.
        ts_computed (int) - The number of t_ij's computed.
        """
        preEv = self.E[:,:2*self.en].dot(self.rev)
        dv = post_event_volts - preEv
        lendv = len(dv)
        dvm = dv[lendv/2:]  # Change in voltage magnitudes
        dvth = dv[:lendv/2] # Change in voltage phase angles
        prevm = preEv[lendv/2:]
        Efingerprint = np.matrix(np.hstack([dvth*prevm, dvm])).T
        if self.verbose:
            print "Initializing topology error search..."
        start_time = time.clock()
        filter_score_dict = self.get_filter_scores(Efingerprint, test_type)
        mid_time = time.clock()
        score_dict = self.get_scores(Efingerprint, filter_score_dict, 
                                     use_filter=use_filter)
        end_time = time.clock()
        if self.verbose:
            print "Topology error search completed."
            print "Filter time: {0}, Fingerprint time: {1}, Total time: {2}".format(
                    mid_time - start_time, end_time - mid_time, end_time - start_time)
        fraction_ts_computed = float(len(score_dict)-1) / len(filter_score_dict)
        return score_dict, filter_score_dict, fraction_ts_computed

    def get_system_matrix_extension(self, split_einds):
        """ Compute the U matrix for a given contingency as described on page 3 
        of the paper.
       
        U is of the form [0 F^T]^T, where F \in {0, 1}^{n \times 2} indicates 
        the rows of the extended system matrix that constrain the voltages of 
        those nodes splitting away to match the master node. This allows us
        to add a slack variable that gives those nodes a different voltage.
        The matrix U^T adds extra constraints to the system that ensures no
        power injection is shared across open breakers in the splitting
        substation.

        Inputs:
        split_einds (List of ints) - The extended indices of the nodes that are
           splitting in this contingency.

        Output:
        A csc_matrix U.
        """
        ind_offset = (self.dHdv.shape[0] + len(self.powernet.pv_inds) + 
                      2*len(self.powernet.slack_inds))
        U_rows = zip(*[self.get_C2_columns(sei) for sei in split_einds])
        U_rows = (ind_offset + np.array(U_rows[0]), ind_offset + np.array(U_rows[1]))
        num_split = len(split_einds)
        row_ind = np.hstack([U_rows[0], U_rows[1]])
        col_ind = np.hstack([np.zeros(num_split), np.ones(num_split)])
        data = np.ones(2*num_split)
        return sps.csc_matrix((data, (row_ind, col_ind)), 
                              shape=(self.A.shape[0], 2))
                              
    def get_subspace(self, split_einds):
        """ Get the filtering fingerprint subspace for this contingency.

        Inputs:
        split_einds (List of ints) - The extended indices of the nodes that are
           splitting in this contingency.
        
        Output:
        sps.csc_matrix. The two-column matrix $\overline{E} A^{-1} U$.
        """
        U = self.get_system_matrix_extension(split_einds)
        subspace = self.EAinv * U
        # Some subspace matrices are singular. Remove columns from those
        # so that they are no longer singular. This step may not be necessary,
        # as I think the lstsq functions tend to be cool with singular matrices.
        while (np.linalg.svd(subspace, compute_uv=False)[-1] < 1e-9 and 
               subspace.shape[1] > 1):
            subspace = subspace[:, :-1]
        return subspace
                
    def get_filter_scores(self, Efingerprint, test_type):
        """ Compute the filter scores tau.

        Run through each possible contingency and compute the filter score
        tau for that contingency. Create a filter_score_dict with the results.

        Inputs:
        Efingerprint (np.matrix) - The voltage change fingerprint.
        test_type ("Full", "Single_Lines") - The type of contingencies for 
           which to test.

        Output:
        filter_score_dict (Dictionary) - A dictionary with contingencies as keys
           and tau scores as values. "No contingency" is not included in this 
           dictionary.
        """
        filter_score_dict = dict()
        filter_scores_computed = 0
        for sub in self.substations:
            for splitting_nodes in sub.node_iterator(test_type):
                split_einds = np.array([sub.nodes[sn].eind for 
                                        sn in splitting_nodes])
                key = (sub.index, tuple(splitting_nodes))
                    
                subspace = self.get_subspace(split_einds)
                x = precision_lstsq(subspace, Efingerprint, ITERATIONS = 1)                
                Efing_sub = subspace.dot(x)
                tau_score = np.linalg.norm(Efingerprint - Efing_sub)
                filter_score_dict[key] = tau_score
                filter_scores_computed += 1
        return filter_score_dict
                    
    def get_scores(self, Efingerprint, filter_score_dict, ts_to_keep = 1, 
                   use_filter = True):
        """ Compute the actual scores t.

        Inputs:
        Efingerprint (np.matrix) - The voltage change fingerprint.
        filter_score_dict (Dictionary) - A dictionary with contingencies as keys
           and tau scores as values. "No contingency" is not included in this 
           dictionary.
        ts_to_keep (int) - As we iterate over possible contingencies, we only
           compute its t score if its filter (tau) score is below the 
           ts_to_keep'th lowest t_ij computed so far. A higher number increases
           computational cost but lowers the chances of filtering out the 
           correct answer. The paper only discusses the default value of 1.
        use_filter (True, False) - Whether or not to filter based on filter 
           (tau) scores. Default is True.
        """
        sort_filtersc = sorted(filter_score_dict.items(), key = lambda x : x[1])
        min_ts = np.array([np.Inf]*ts_to_keep)
        
        score_dict = dict()
        score_dict[-1] = np.linalg.norm(Efingerprint) # Represents "no topology
                                                      # change."
        for i in xrange(len(sort_filtersc)):
            key, tau = sort_filtersc[i]
            splitbus, splitting_nodes = key
            if (filter_score_dict[key] - np.max(min_ts) > 0) and use_filter:
                continue
            split_einds = np.array([self.substations[splitbus].nodes[sn].eind 
                                    for sn in splitting_nodes])
            U = self.get_system_matrix_extension(split_einds)
            FT = U[-len(self.lamed):,:].T
            UAinvU = U.T * self.Afac(U.todense())
            try:
                gamma = np.linalg.solve(UAinvU, FT * self.lamed)
            except np.linalg.LinAlgError:
                gamma = precision_lstsq(UAinvU, FT * self.lamed, ITERATIONS = 1)
            Efing_approx = -self.EAinv * (U * gamma)
            t_score = np.linalg.norm(Efingerprint - Efing_approx)
            score_dict[key] = t_score
            if t_score < np.max(min_ts):
                min_ts[np.argmax(min_ts)] = t_score
        return score_dict
