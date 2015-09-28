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
	
	Copyright Colin Ponce 2014.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import math, cmath
import sys
from itertools  import chain
from Bus import Bus
from Branch import Branch
from Power_Utils import powerflow, powerflow_jacob

class PowerNetwork:
    #PUBLIC METHODS
    
    """
    Create a PowerNetwork. Note that all attributes of buses and edges are listed
    in the IEEE CDF format documentation at
    http://www.ee.washington.edu/research/pstca/formats/cdf.txt.    
    """
    def __init__(self, buses_byID, branches):
        self.branches = branches
        self.buses_byID = buses_byID
        self.nb = len(buses_byID)
        self.buses = sorted(self.buses_byID.values(), key = lambda x : x.index)
        self.types = [bus.type for bus in self.buses]
        self.slack_inds = [bus.index for bus in self.buses if bus.type == 'Slack']
        self.pv_inds = [bus.index for bus in self.buses if bus.type == 'PV']

        Y, dc_mat = PowerNetwork.create_Y(self.buses, self.branches, create_DC = True)
        self.Y = Y
        self.dc_mat = dc_mat
        
        self.init_guess = np.hstack([[np.angle(bus.voltage) for bus in self.buses],
                                     [  np.abs(bus.voltage) for bus in self.buses]])
                                     
                                     
    def prep_for_delete(self):
        for bus in self.buses:
            bus.branches = None
        for branch in self.branches.values():
            branch.buses = None


    @staticmethod
    def create_Y(buses, branches, create_DC=False):
        nb = len(buses)
        Y = sps.lil_matrix((nb, nb), dtype=complex)
        if create_DC:
            dc_mat = sps.lil_matrix((nb, nb), dtype=np.float64)
        for branch in branches.values():
            bus0, bus1 = [bus.index for bus in branch.buses]
            admittance = branch.admittance
            if branch.nonunity_tap:
                tap = branch.tap
            else:
                tap = 1.0
            Yii = (admittance + 1j*branch.line_charging_susc / 2.0)
            Y[bus0, bus0] += Yii / (tap*np.conj(tap))
            Y[bus1, bus1] += Yii
            Y[bus0, bus1] -= admittance / np.conj(tap)
            Y[bus1, bus0] -= admittance / tap

            if create_DC:
                dc_approx_weight = - 1.0 / np.imag(branch.impedance)
                dc_mat[bus0, bus1] -= dc_approx_weight
                dc_mat[bus1, bus0] -= dc_approx_weight
                dc_mat[bus0, bus0] += dc_approx_weight
                dc_mat[bus1, bus1] += dc_approx_weight
                
        for bus in buses:
            Y[bus.index, bus.index] += bus.shunt_adm
        
        if create_DC:
            return Y.tocsr(), dc_mat.tocsr()
        else:
            return Y.tocsr()

            
          
    def voltages(self):
        return np.array([bus.voltage for bus in self.buses])
        
    def injection_targets(self):
        return np.array([bus.generation - bus.load for bus in self.buses])
        
    def copy(self):
        """
        Create a copy of this PowerNetwork.
        """
        new_buses_byID = dict()
        for id, bus in self.buses_byID.iteritems():
            new_buses_byID[id] = bus.copy()
        new_branches = dict()
        for id, branch in self.branches.iteritems():
            new_branches[id] = branch.copy()
        return PowerNetwork(new_buses_byID, new_branches)

    @staticmethod    
    def get_observable_projection_matrix(Y, pmus):
        """
        Get the matrix E tha projects a voltage vector onto the components rendered
        observable by a set of PMUs.
        
        Inputs:
        Y - The network admittance matrix.
        pmus - A list of the nodes with PMUs on them.
        
        Outputs:
        E - The projection matrix.
        """
        obs_buses = np.array([])
        for pmu in pmus:
            from_here = np.nonzero(Y[:,pmu])[0]
            obs_buses = np.concatenate([obs_buses, from_here])
        obs_buses = list(set(obs_buses))
        obs_buses = sorted(obs_buses)
        
        nb = Y.shape[0]
        ob = len(obs_buses)
        E = np.zeros([2*ob, 2*nb])
        for i in range(ob):
            E[i, obs_buses[i]] = 1
            E[ob + i, nb + obs_buses[i]] = 1
        return E
    
    def Y_to_gml(self, filename, bus_type, lines_to_highlight,
                       buses_to_highlight):
        """
        Convert an admittance matrix with coloring information to GML format
        for graph visualiation. You can learn about the GML format at
        http://en.wikipedia.org/wiki/Graph_Modelling_Language
        or
        http://gephi.github.io/users/supported-graph-formats/gml-format/
        
        Inputs:
        filename - The name of the GML file to which to print.
        bus_type - A vector indicating the type of each bus.
                   0 indicates PQ bus.
                   2 indicates PV bus.
                   3 indicates Vtheta bus.
        lines_to_highlight - For each branch, contains either False or the text
                             hexadecimal value indicating the red component of
                             the color with which to draw the line.
        buses_to_highlight - For each bus, contains either False or the text
                             hexadecimal value indicating the red component of
                             the color with which to draw the line.
        """
        f = open(filename, 'w')
        f.write("graph [\n")
        f.write("    id 0\n")
        f.write("    version 0\n")
        f.write("    directed 0\n")
        for k in range(self.num_buses):
            f.write("    node [\n")
            f.write("        id {0}\n".format(k+1))
            f.write("        label \"Bus {0}\"\n".format(k+1))
            f.write("        graphics [\n")
            if bus_type[k] == 'PV':
                f.write("            type \"ellipse\"\n")
            elif bus_type[k] == 'Slack':
                f.write("            type \"rectangle\"\n")
            else:
                f.write("            type \"triangle\"\n")
            if type(buses_to_highlight[k]) == str:
                f.write("            fill \"#00" + buses_to_highlight[k] + "00\"\n")
            elif buses_to_highlight[k]: 
                f.write("            fill \"#00FF00\"\n")
            f.write("            w 120.0 \n")
            f.write("            h 120.0 \n")
            f.write("            ]\n")
            f.write("    ]\n")
        for i in range(len(lines_to_highlight)):
            branch = self.branches[i]
            branch_imp = branch['resistance'] + 1j * branch['reactance']
            branch_adm = 1 / branch_imp
            f.write("    edge [\n")
            f.write("        source {0}\n".format(branch['buses'][0]+1))
            f.write("        target {0}\n".format(branch['buses'][1]+1))
#            f.write("        label \"{0:.3}\"\n".format(edge_adm))
            f.write("        graphics [\n")
            if lines_to_highlight[i]:                    
                f.write("            fill \"#" + lines_to_highlight[i] + "0000\"\n")
                f.write("            width 8\n")
            else:
                f.write("            fill \"#000000\"\n")
                f.write("            width 2\n")
            f.write("        ]\n")
            f.write("    ]\n")
        f.write("]\n")
        f.close()

    @classmethod
    def powernet_from_IEEECDF(cls, filename):
        """Read in a power network from the IEEE Common Data Format.
        
        More fully, it's the IEEE Common Data Format for the Exchange of 
        Solved Load Flow Data. You can learna about this format at
        http://www.ee.washington.edu/research/pstca/
        
        Arguments:
        filename -- The name of the f containing the data.
        
        Returns:
        A PowerNetwork instance.
        
        """
        assert isinstance(filename, str)
        f = open(filename, 'r')
        line = f.readline()
        line = line.strip()
        MVA_base = float(line[30:37])
        f.readline()
        line = f.readline()
        buses_byID = dict()
        
        bustypes = {0 : 'PQ', 1 : 'PQ', 2 : 'PV', 3 : 'Slack'}
        
        # Read in bus specs
        bus_ind = 0
        while line[0:4] != "-999":
            new_bus_ind = bus_ind
            ID = int(line[0:4]) - 1
            
            name = line[7:17].strip()
            type = bustypes[int(line[24:26])]
            finalmag = float(line[27:33])
            finalangle = float(line[33:40]) * 2*math.pi / 360
            loadMW = float(line[40:49]) / MVA_base
            loadMVAR = float(line[49:59]) / MVA_base
            genMW = float(line[59:67]) / MVA_base
            genMVAR = float(line[67:75]) / MVA_base
            desiredV = float(line[84:90])
            shuntcond = float(line[106:114])
            shuntsusc = float(line[114:122])
            
            load = loadMW + 1j * loadMVAR
            gen = genMW + 1j*genMVAR
            shunt_adm = shuntcond + 1j*shuntsusc
            finalvoltage = finalmag * np.exp(1j * finalangle)
            buses_byID[ID] = Bus(new_bus_ind, ID, load, gen, type, shunt_adm, finalvoltage)
            
            bus_ind += 1
            line = f.readline()
        line = f.readline()
        line = f.readline()
        branches = dict()
        # Read in line info
        branch_ind = 0
        while line[0:4] != "-999":
            ID = branch_ind
            busIDs = (int(line[0:4]) - 1, int(line[5:9]) - 1)
            branch_buses = (buses_byID[busIDs[0]], buses_byID[busIDs[1]])
            
            type = int(line[18])
            resistance = float(line[19:29])
            reactance = float(line[29:40])
            impedance = resistance + 1j * reactance
            line_charging_susc = float(line[40:50])
            tap_ratio = float(line[76:82])
            tap_shift = float(line[83:90])
            tap = tap_ratio * np.exp(1j * np.pi/180 * tap_shift)
            nonunity_tap = tap != 0.0
            branches[ID] = Branch(ID, branch_buses, impedance, 
                                  line_charging_susc, nonunity_tap, tap)
            branch_ind += 1
            line = f.readline()
            
        return cls(buses_byID, branches)
        
    @classmethod
    def powernet_from_matpower(cls, filename):
        assert isinstance(filename, str)
        f = open(filename, 'r')
        
        bustypes = {1 : 'PQ', 2 : 'PV', 3 : 'Slack'}
        
        line = f.readline().strip()
        while not line.startswith("mpc.baseMVA"):
            line = f.readline().strip()
        i1, i2 = (line.find('=')+1, line.find(';'))
        MVA_base = float(line[i1 : i2])
        
        line = f.readline().strip()
        while not line.startswith("mpc.bus"):
            line = f.readline().strip()
        buses = dict()
        line = f.readline().strip()
        bus_ind = 0
        while not line.startswith("];"):
            tokens = line.split()
            new_bus = dict()
            new_bus_ind = bus_ind
            ID = int(tokens[0]) - 1
            name = tokens[0]
            type = bustypes[int(tokens[1])]

            loadMW = float(tokens[2]) / MVA_base
            loadMVAR = float(tokens[3]) / MVA_base
            gen = 0.0 + 0.0j
            shuntcond = float(tokens[4]) / MVA_base
            shuntsusc = float(tokens[5]) / MVA_base
            shunt_adm = shuntcond + 1j * shuntsusc
            finalmag = float(tokens[7])
            finalangle = float(tokens[8]) * 2 * math.pi / 360.0
            finalvoltage = finalmag * np.exp(1j * finalangle)
            
            load = loadMW + 1j * loadMVAR
            buses[ID] = Bus(new_bus_ind, ID, load, gen, type, shunt_adm, finalvoltage)
            bus_ind += 1
            line = f.readline().strip()
        
        while not line.startswith("mpc.gen"):
            line = f.readline().strip()
        line = f.readline().strip()
        while not line.startswith("];"):
            tokens = line.split()
            genMW = float(tokens[1]) / MVA_base
            genMVAR = float(tokens[2]) / MVA_base
            buses[int(tokens[0])-1].generation = genMW + 1j * genMVAR
            line = f.readline().strip()
        
        while not line.startswith("mpc.branch"):
            line = f.readline().strip()
        line = f.readline().strip()
        branches = dict()
        branch_ind = 0
        while not line.startswith("];"):
            tokens = line.split()
            if tokens[10] == 0:
                continue
            ID = branch_ind
            busIDs = (int(tokens[0]) - 1, int(tokens[1]) - 1)
            branch_buses = (buses[busIDs[0]], buses[busIDs[1]])
            resistance = float(tokens[2])
            reactance = float(tokens[3])
            impedance = resistance + 1j * reactance
            line_charging_susc = float(tokens[4])
            tap_ratio = float(tokens[8])
            tap_shift = float(tokens[9])
            tap = tap_ratio * np.exp(1j * np.pi/180 * tap_shift)
            nonunity_tap = tap != 0.0
            branches[ID] = Branch(ID, branch_buses, impedance, line_charging_susc, nonunity_tap, tap)
            
            branch_ind += 1
            line = f.readline().strip()

        return cls(buses, branches)

    
    """
    Compute the number of hops along the network between two branches.
    
    Inputs:
    branch1 - One of the two branches.
    branch2 - The other branch.
    
    Output: Number of hops between the two branches.
    """
    def distance_between_branches(self, branch1, branch2):
        # Construct dual graph
        branch1_ind = -1
        branch2_ind = -1
        dual_top = np.zeros((len(self.branches), len(self.branches)))
        for i in range(len(self.branches)):
            if self.branches[i].ID == branch1.ID:
                branch1_ind = i
            if self.branches[i].ID == branch2.ID:
                branch2_ind = i
            iBus1, iBus2 = [bus.index for bus in self.branches[i].buses]
            for j in range(i+1, len(self.branches)):
                jBus1, jBus2 = [bus.index for bus in self.branches[j].buses]
                if iBus1 == jBus1 or iBus1 == jBus2 or iBus2 == jBus1 or iBus2 == jBus2:
                    dual_top[i,j] = 1
                    dual_top[j,i] = 1
        
        # Breadth first search to find distance between branches
        BFS_vec = np.zeros(dual_top.shape[0])
        BFS_vec[branch1_ind] = 1
        steps = 0
        while BFS_vec[branch2_ind] == 0:
            BFS_vec = np.dot(dual_top, BFS_vec)
            steps += 1
        return steps 


    def get_neighbors(self, i):
        """ Get a list of a bus's neigbhors.

        Inputs:
        i - The index of the bus in question.

        Output:
        A list of Bus objects.
        """
        return self.buses[i].get_neighbors()
                

    def remove_branch(self, branch):
        """Remove a power line from the power network.
        
        Note that the branch is still part of the list of network branches. This merely
        removes it from the network's admittance matrix.
        
        Arguments:
        branch - The branch to remove from the admittance matrix.
        """            
        PowerNetwork.remove_line_from_Y(self.Y, branch)
        
    @staticmethod
    def remove_line_from_Y(Y, branch):
        """Remove a line from a power network admittance matrix.
        
        Arguments:
        Y -- The admittance matrix.
        branch - The branch to remove from the admittance matrix.
        
        Returns:
        A new matrix.
        """
        bus0, bus1 = [bus.index for bus in branch.buses]
        admittance = branch.admittance
        line_charging_susc = branch.line_charging_susc
        if branch.nonunity_tap:
            tap = branch.tap
        else:
            tap = 1.0
        Y[bus0, bus1] += admittance / np.conj(tap)
        Y[bus1, bus0] += admittance / tap
        Y[bus0, bus0] -= (admittance + 0.5*1j*line_charging_susc) / (tap*np.conj(tap))
        Y[bus1, bus1] -= (admittance + 0.5*1j*line_charging_susc)            



    def add_branch(self, branch):
        PowerNetwork.add_line_to_Y(self.Yrix, branch)

    @staticmethod
    def add_line_to_Y(Y, branch):
        bus0, bus1 = branch.buses
        branch_imp = branch['resistance'] + 1j * branch['reactance']
        branch_adm = 1.0 / branch_imp
        admittance = branch.admittance
        line_charging_susc = branch.line_charging_susc
        if branch.nonunity_tap:
            tap = branch.tap
        else:
            tap = 1.0
        Y[bus0, bus1] -= admittance / np.conj(tap)
        Y[bus1, bus0] -= admittance / tap
        Y[bus0, bus0] += (admittance + 0.5*1j*line_charging_susc) / (tap*np.conj(tap))
        Y[bus1, bus1] += (admittance + 0.5*1j*line_charging_susc)            

    @staticmethod
    def newton_raphson(init, fun, jacob, args, TOL = 1e-10, max_iters = 25):
        """Use Newton's method to find a root of the function fun.
        
        Arguments:
        init -- A sequence of length k that represents an initial guess.
        fun -- The function whose roots we wish to find. This function take a
               sequence of length k as input and returns a sequence of length
               k as output.
        jacob -- A function that computes the Jacobian of fun at point 
                 x. Takes a sequence of length k as input and returns a k x k 
                 np.array as output. 
        args -- A list of extra arguments to be passed to fun() and jacob().
        TOL -- (Optional) Iterations cease when an x is found such that
               ||fun(x)||_2 < TOL.
        max_iters -- (Optional) The number of iterations to perform before 
                     announcing nonconvergence.
        
        Output:
        A sequence of length k that is a root of fun.
        
        """
        init = np.array(init)
        x_curr = np.array(init.copy())
        curr_resid_norm = np.linalg.norm(fun(x_curr, *args))
        iter_num = 0
        while curr_resid_norm >= TOL and iter_num < max_iters:
            # print "Iteration: {0}, resid norm: {1}".format(iter_num, curr_resid_norm)
            iter_num = iter_num + 1
            J = jacob(x_curr, *args)
            try:
                delta_x = spsl.spsolve(J, -fun(x_curr, *args))
            except np.linalg.LinAlgError:
                delta_x = spsl.lsmr(J, -fun(x_curr, *args))[0]
            if np.isnan(np.sum(delta_x)):
                delta_x = spsl.lsmr(J, -fun(x_curr, *args))[0]
            x_old = x_curr.copy()
            old_resid_norm = curr_resid_norm
            x_curr = x_old + delta_x
            curr_resid_norm = np.linalg.norm(fun(x_curr, *args))
            step_size = 2
            # Bisecting line search
            while curr_resid_norm > old_resid_norm:
                step_size = step_size / 2.0
                x_curr = x_old + step_size * delta_x
                curr_resid_norm = np.linalg.norm(fun(x_curr, *args))
        if curr_resid_norm > TOL or np.isnan(np.sum(x_curr)):
            raise RuntimeError("PowerNetwork.newton_raphson: Non-convergence.")
        else: return x_curr
        
    def simulate(self):
        nb = self.nb
        slack_inds = self.slack_inds
        pv_inds = self.pv_inds
        
        vS = np.array([(np.angle(x), np.abs(x)) for x in [self.buses[i].voltage for i in slack_inds]]).flatten()
        vP = np.array([np.abs(self.buses[i].voltage) for i in pv_inds])
        injection_targets = self.injection_targets()
        sreal, sreactive = np.real(injection_targets), np.imag(injection_targets)
        s = np.hstack([vS, vP, sreal, sreactive])
        
        args = (self.Y, pv_inds, slack_inds, s)
        
        init_guess = np.hstack([self.init_guess, np.zeros(len(vS)+len(vP))])
        X = PowerNetwork.newton_raphson(init_guess, PowerNetwork.system_function,
                                        PowerNetwork.system_jacobian, args, TOL=1e-9)
        V = X[nb:2*nb] * np.exp(1j * X[:nb])
        for i in xrange(self.nb):
            self.buses[i].voltage = V[i]
        return V
        
    @staticmethod
    def get_C(pv_inds, slack_inds, nb):
        nC = len(pv_inds) + 2*len(slack_inds)
        C = sps.lil_matrix((nC, 2*nb))
        ind = 0
        for i in slack_inds:
            C[ind  , i   ] = 1
            C[ind+1, i+nb] = 1
            ind += 2
        for i in pv_inds:
            C[ind  , i+nb] = 1
            ind += 1
        return C

    @staticmethod
    def system_function(x, Y, pv_inds, slack_inds, s):
        nb = Y.shape[0]
        vX = x[:2*nb]
        pF = powerflow(vX,Y)
        C = PowerNetwork.get_C(pv_inds, slack_inds, nb)
        pF += C.T.dot(x[2*nb:])
        out = np.hstack([C.dot(x[:2*nb]), pF]) - s
        return out

    @staticmethod
    def system_jacobian(x, Y, pv_inds, slack_inds, s_bus):
        nb = Y.shape[0]
        pJ = powerflow_jacob(x[:2*nb], Y)
        C = PowerNetwork.get_C(pv_inds, slack_inds, nb)            
        J = sps.bmat([[ C, None],
                      [pJ,  C.T]], format='csr')
        return J

if __name__=="__main__":
    filename = sys.argv[1]
    powernet = PowerNetwork.powernet_from_IEEECDF(filename)
    powernet.simulate()
    print powernet.voltages()
