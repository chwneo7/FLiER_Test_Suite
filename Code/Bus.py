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
import Branch

"""
This file establishes Bus and Substation objects in a network.

Terminology: A "substation" represents a bus in a bus/branch power 
system model. A "node" represents a busbar section within a substation in a 
node/breaker model. A "bus" refers either one depending on the model level in 
use; it is a vertex in the power network graph.
"""
        
class Bus(object):
    """ABSTRACT CLASS. A Bus is a single element in a network model.
    
    What the Busbar represents depends on the level of network model. In a 
    bus/branch model, the Busbar object represents a Ring_Substation. In a 
    node/breaker model, the Busbar object represents a Node.
    """

    def __init__(self, index, ID, load = 0.0, generation = 0.0, type = 'PQ', 
                 shunt_adm = 0.0+0.0j, voltage=1.0+0.0j, branches = None):
        self.index = index
        self.ID = ID
        self.load = load
        self.generation = generation
        self.type = type
        self.shunt_adm = shunt_adm
        self.voltage = voltage
        if branches is None:
            self.branches = []
        else:
            self.branches = branches
            
    def copy(self):
        return Bus(self.index, self.ID, self.load, self.generation, self.type,
                   self.shunt_adm, self.voltage, self.branches)
                
    def add_branch(self, branch):
        self.branches.append(branch)
        
    def get_neighbors(self):
        """Get the Bus objects to which this Bus is connected."""
        return [(branch.buses[0] if branch.buses[0] != self else branch.buses[1]) for branch in self.branches]

class Ring_Substation(Bus):
    """A Ring_Substation is a type of bus.

    The Ring_Substation is initialized as a substation with no connections to
    other substations. That is, self.isolated is set to True.
    
    The Ring_Substation models a substation as nodes connected by circuit 
    breakers in a ring arrangement. In this implementation, we model substations
    as having all generation taking place on Node 0, all load and shunt 
    admittance taking place on Node 1, and all other Nodes having zero injection
    but having a single branch to a Node on another Ring_Substation.

    One could in theory model the breakers within a substation as branches in 
    the network. However, they will typically have exceedingly high admittances,
    resulting in unstable computations. Instead, we model them by adding variables
    that allow us to shift power injection among Nodes in a substation, as well as 
    extra constraints that ensure all Nodes in a substation have the same voltage.
    This technique is discussed in the paper on FLiER.
    """

    class Node(Bus):
        """A node in a node/breaker model."""
        def __init__(self, substation, index, load = 0.0, generation = 0.0, 
                     type = 'PQ', shunt_adm = 0.0+0.0j, voltage=1.0+0.0j, 
                     branch = None, opposing_node = None):
            branches = [branch] if branch is not None else None
            Bus.__init__(self, index, index, load, generation, type, shunt_adm, 
                         voltage, branches)
             
            self.substation = substation
            # if branch is not None:
            #     self.branch = branch
            # else:
            #     self.branch = None
            if opposing_node is not None:
                self.opposing_node = opposing_node
            else:
                self.opposing_node = None
                
            self._eind = None

        @property
        def branch(self):
            assert len(self.branches) <= 1
            if len(self.branches) == 0:
                return None
            else:
                return self.branches[0]
            
        @property
        def eind(self):
            assert self._eind is not None
            return self._eind
            
        @eind.setter
        def eind(self, value):
            self._eind = value
                        
    GEN_IND = 0
    LOAD_IND = 1
    
    def __init__(self, bus):
        """Create a new Ring_Substation from a Bus object.

        Inputs:
        bus (Bus) - The bus/branch-level bus object.
        """
        Bus.__init__(self, bus.index, bus.ID, bus.load, bus.generation, 
                     bus.type, bus.shunt_adm, bus.voltage, bus.branches)
        Node = Ring_Substation.Node
        GEN_IND, LOAD_IND = Ring_Substation.GEN_IND, Ring_Substation.LOAD_IND
                  
        self.nodes = []
        self.nodes.append(Node(self, GEN_IND, generation = self.generation, 
                               type=self.type, voltage=self.voltage))
        self.nodes.append(Node(self, LOAD_IND, load = self.load, 
                               shunt_adm = self.shunt_adm, voltage=self.voltage))
        # To have a ring we need at least three nodes. So, create a third node
        # that is zero-injection and not attached to anything.
        self.nodes.append(Node(self, 2, voltage = self.voltage))
        self._branch2node = dict()
        self.isolated = True
        
    def num_nodes(self):
        return len(self.nodes)
        
    def set_extended_indices(self, starting_index):
        """Given a starting index, set the extended, node-level indices for
        all nodes at this Ring_Substation.
        """
        curr_index = starting_index
        for node in self.nodes:
            node.eind = curr_index
            curr_index += 1
        
    def add_node(self, branch = None, opposing_node = None):
        """Add a new zero-injection node to this Ring_Substation.

        Inputs:
        branch (Branch) - A branch connecting this Node to another.
        opposing_node (Node) - The Node on the other end of the branch.
        """
        Node = Ring_Substation.Node
        if self.isolated:
            # If currently isolated, replace the current third Node with a
            # new one.
            self.nodes[-1] = Node(self, self.nodes[-1].index,
                                  voltage = self.voltage, branch = branch,
                                  opposing_node = opposing_node)
            self.isolated = False
        else:
            self.nodes.append(Node(self, self.nodes[-1].index+1,
                                   voltage = self.voltage, branch = branch,
                                   opposing_node = opposing_node))
        
    @staticmethod
    def connect_nodes(substation0, substation1, branch):
        """Connect two Ring_Substations via a branch.

        Inputs:
        substation0 (Ring_Substation) - One end of the branch.
        substation1 (Ring_Substation) - The other end of the branch.
        branch (Branch) - The branch between the two substations.
        """
        # The opposing node doesn't yet exist, so can't pass it as a parameter.
        substation0.add_node(branch, opposing_node = None)
        substation1.add_node(branch, opposing_node = substation0.nodes[-1])
        # Now it exists, so fix it.
        substation0.nodes[-1].opposing_node = substation1.nodes[-1]
        substation0._branch2node[branch] = substation0.nodes[-1]
        substation1._branch2node[branch] = substation1.nodes[-1]
        
    def branch_at_node(self, i):
        """Return the branch connected to a node of this Ring_Substation.

        Inputs:
        i (int) - The Node index.

        Output: A Branch.
        """
        assert len(self.nodes[i].branches) < 2
        if self.nodes[i].branch is None:
            return None
        else: return self.nodes[i].branch
        
    def branchID_at_node(self, i):
        """Return the ID of the branch connected to a Node.

        Inputs:
        i (int) - The Node index.

        Output: An int.
        """
        branch = self.branch_at_node(i)
        if branch is None:
            return None
        else: return branch.ID
        
    def branch2node(self, branch):
        """Return the Node associated with a given Branch."""
        return self._branch2node[branch]
        
    def node_across_branch(self, i):
        """Get the Node at the other end of the branch attached to this Node.

        Inputs:
        i (int) - The Node index.

        Output: A Node.
        """
        return self.nodes[i].opposing_node
        
    def substation_across_branch(self, i):
        """Return the opposing Ring_Substation at a Node.

        Inputs:
        i (int) - The Node index.

        Output: A Ring_Substation.
        """
        nab = self.node_across_branch(i)
        if nab is None:
            return None
        else:
            return self.node_across_branch(i).substation
        
    def node_iterator(self, test_type='Sub_Split'):
        """Return an iterator to cycle through contingencies at this substation.
        """
        return self.Node_Set_Iterator(self, test_type)

        
    class Node_Set_Iterator(object):
        """Node_Set_Iterator

        The Node_Set_Iterator cycles through possible contingencies associated 
        with this Ring_Substation. There are two types of iterations:

        Full - This iterator cycles through all contingencies in which the ring
           of the substation splits in two, creating two electrically distinct
           "line" substations.
        Single_Lines - Thie iterator cycles through contingencies in which a 
           single node connected to a branch splits away. This creates a zero-
           injection leaf node in the network, effectively removing the line
           from the network.
        """
        def __init__(self, sub, type='Full'):
            """Create a new Node_Set_Iterator.

            Inputs:
            sub (Ring_Substation) - The Ring_Substation through whose 
               contingencies this iterator will iterate.
            type ("Full", "Single_Lines") - The type of iteration to perform.
               Default is "Full".
            """
            self.sub = sub
            self.num_nodes = len(sub.nodes)
            if type == 'Full':
                self.node1 = 1
                self.node2 = 1
                FIRST_BRANCH_NODE = 2
                self.increment_condition = self.full_increment_condition
            elif type == 'Single_Lines':
                self.node1 = 2
                self.node2 = 1
                # self.increment_condition = lambda n1, n2 : n1 != n2
                self.increment_condition = self.single_line_increment_condition
            else:
                assert False
                
        def full_increment_condition(self, n1, n2):
            """Check if the current set of nodes to split is a valid contingency.

            Checks for a full contingency.

            Inputs:
            n1 (int) - The first node around the ring that would split away.
            n2 (int) - The last node around the ring that woud split away.

            Output: A boolean. If True, it is not a valid contingency, so
               continue incrementing.
            """
            FIRST_BRANCH_NODE = 2
            if n1 == Ring_Substation.LOAD_IND and self.sub.load == 0:
                # Zero load, so the load node does nothing.
                return True
            if n1 <= FIRST_BRANCH_NODE and n2 == self.num_nodes - 1:
                # Pulls away all branches, so either it does nothing or it
                # creates an unsolvable case.
                return True
            if n1 == Ring_Substation.LOAD_IND:
                # Guaranteed nonzero load, so pulling away the load node
                # does something unique.
                return False
            if len(set([self.sub.substation_across_branch(i).index for 
                        i in np.arange(n1, n2+1)])) == 1:
                # This is the "line failure" equivalent case.
                # Usually this is because the splitting set is a single node, but
                # not always. It means that there is a single bus on the other
                # end of the splitting branch set. The load node is not present,
                # as guaranteed by the previous case. This equivalent case will 
                # show up twice, once here and once on the other bus. So pick
                # one and skip the other.
                if self.sub.index > self.sub.substation_across_branch(n1).index:
                    return True
            if (n2 == self.num_nodes - 1 and self.sub.generation == 0 and 
                self.sub.type == 'PQ'):
                # The set [1,2,...,k] is equivalent to the set [k+1,k+2,...m]. So
                # if m = self.num_nodes - 1 is present, skip.
                return True
            return False
            
        def single_line_increment_condition(self, n1, n2):
            """Check if the current set of nodes to split is a valid contingency.

            Checks for a single-line contingency.

            Inputs:
            n1 (int) - The first node around the ring that would split away.
            n2 (int) - The last node around the ring that woud split away.

            Output: A boolean. If True, it is not a valid contingency, so
               continue incrementing.
            """
            if n1 != n2:
                return True
            if self.sub.index > self.sub.bus_across_branch(n1).index:
                return True
            return False
            
                
        def __iter__(self):
            return self
            
        def increment(self):
            self.node2 += 1
            if self.node2 == self.num_nodes:
                self.node1 += 1
                self.node2 = self.node1
                
        def next(self):
            """Get the next possible contingency."""
            self.increment()
            if self.node1 == self.num_nodes:
                raise StopIteration
            while self.increment_condition(self.node1, self.node2):
                self.increment()
                if self.node1 == self.num_nodes:
                    raise StopIteration
            return np.arange(self.node1, self.node2+1)
