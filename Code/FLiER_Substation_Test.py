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
from FLiER_Substation import FLiER_Substation
from Bus import Bus
import Power_Utils
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle

def split_bus(powernet, sub, splitting_nodes):
    """Create a new Power_Network with a split substation.

    Inputs:
    powernet (Power_Network) - The original power network.
    sub (Ring_Substation) - The substation that will split.
    splitting_nodes (List of ints) - The substation nodes that will split from
       the substation.

    Output:
    Power_Network. The new power network that results from the substation 
    splitting.
    """
    new_buses = {bus.ID : bus.copy() for bus in powernet.buses}
    newbusID = max([bus.ID for bus in powernet.buses]) + 1
    newbusind = len(powernet.buses)
    new_bus = Bus(newbusind, newbusID, voltage=sub.voltage)
    new_buses[newbusID] = new_bus
    old_bus = new_buses[sub.ID]
    for node in splitting_nodes:
        new_bus.load += sub.nodes[node].load
        old_bus.load -= sub.nodes[node].load

        new_bus.generation += sub.nodes[node].generation
        old_bus.generation -= sub.nodes[node].generation
        
        new_bus.shunt_adm += sub.nodes[node].shunt_adm
        old_bus.shunt_adm -= sub.nodes[node].shunt_adm
    
    splitting_branch_IDs = [sub.branchID_at_node(sn) for sn in splitting_nodes
                            if sub.branchID_at_node(sn) is not None]
    new_branches = {branch.ID : branch.copy() for branch 
                                                in powernet.branches.values()}
    for branch_ID in splitting_branch_IDs:
        branch = new_branches[branch_ID]
        assert len(branch.buses) == 2
        if branch.buses[0].ID == old_bus.ID:
            branch.buses = (new_bus, branch.buses[1])
        elif branch.buses[1].ID == old_bus.ID:
            branch.buses = (branch.buses[0], new_bus)
        else: assert False
        
    return PowerNetwork(new_buses, new_branches)
    
def get_post_event_volts(sub, splitting_nodes, powernet, curr_FLiER):
    """Simulate a contingency and return the resulting network voltages.

    Inputs:
    sub (Ring_Substation) - The substation where the contingency occurs.
    splitting_nodes (List of ints) - The set of nodes splitting from the 
       substation.
    powernet (Power_Network) - The power network.
    curr_FLiER (FLiER_Substation) - The FLiER object that will run the FLiER
       algorithm.

    Output:
    List of 2n doubles. The set of post-contingency voltages. Angles followed
    by magnitudes.
    """
    split_powernet = split_bus(powernet, sub, splitting_nodes)
    V_split = split_powernet.simulate()
    new_bus_volt = V_split[-1]
    
    eVs = FLiER_Substation.extend_complex_vector(curr_FLiER.substations, V_split[:-1],
                                                  curr_FLiER.en)
    eVs = np.hstack([np.angle(eVs), np.abs(eVs)])
    for sn in splitting_nodes:
        eVs[sub.nodes[sn].eind] = np.angle(new_bus_volt)
        eVs[sub.nodes[sn].eind+curr_FLiER.en] = np.abs(new_bus_volt)
    eVs = eVs[curr_FLiER.reindr2eindr]
    post_event_volts = curr_FLiER.E[:,:2*curr_FLiER.en].dot(eVs)
    split_powernet.prep_for_delete()
    return post_event_volts
    
def write_contingency_results(file, c_sub, c_splitting_nodes, score = None):
    """Write a contingency to file.

    Inputs:
    c_sub (Ring_Substation) - The substation associated with the contingency.
    c_splitting_nodes (List of ints) - The substation nodes that split in the contingency.
    score - (Optional) The score that FLiER associated with this contingency.
    """
    # Set of nodes splitting from the substation.
    node_str = ", ".join([str(node) for node in c_splitting_nodes])
    # Set of neighborind substations connected to those nodes that split.
    nbrs_str = ", ".join([str(c_sub.substation_across_branch(j).index) for j
                          in c_splitting_nodes if 
                          c_sub.node_across_branch(j) is not None])
    if score is None:
        file.write("{0}; {1}; {2}\n".format(str(c_sub.index), node_str, 
                                                                nbrs_str))
    else:
        file.write("{0}; {1}; {2}; {3}\n".format(str(c_sub.index), node_str, 
                                                    nbrs_str, str(score)))
                                                
def write_to_file(file, solution_key, flier_output, substations):
    """ Write a test result to disk.

    *Contingencies have the following format: A tuple.
       Element 0: The index of the substation that split.
       Element 1: The set of nodes of that substation that split away.
    Note that single line failures can be described as the event (sub, node),
    where sub is one of the substations attached to the failed line and node is
    the node attached to the line that failed.

    Inputs:
    file - The file object to which to write.
    solution_key - The correct event.*
    flier_output - A list of the data output by FLiER. Format of each element is
       tuple:
       Element 0 (scores): A dictionary of contingencies* and the scores t_ij
          that FLiER assigned to each.
       Element 1 (filter_scores): A dictionary of contingencies* and the filter
          scores tau_ij that the FLiER filter assigned to each.
    substations - A dictionary of substations with substation indices as their
       keys.
    """
    split_bus_ind, splitting_nodes = solution_key
    scores, filter_scores, fraction_ts_computed = flier_output
    sub = substations[split_bus_ind]
    write_contingency_results(file, sub, splitting_nodes)
    file.write("{0}\n".format(str(fraction_ts_computed)))
    
    for key, val in scores.iteritems():
        if key == -1:
            file.write("{0};  ;   ; {1}\n".format(str(key), str(val)))
            continue
        test_sub = substations[key[0]]
        test_splitting_nodes = key[1]
        write_contingency_results(file, test_sub, test_splitting_nodes, val)
    file.write("\n")
    for key, val in filter_scores.iteritems():
        if key == -1:
            file.write("{0};  ;   ; {1}\n".format(str(key), str(val)))
        else:
            test_sub = substations[key[0]]
            test_splitting_nodes = key[1]
            write_contingency_results(file, test_sub, test_splitting_nodes, val)
    file.write("END TEST\n\n")
    
def read_inputs(args):
    """Read in test parameters.

    See the main() function for a list of possible parameters.

    Output: A tuple containing the given parameters.
    """
    filename = None
    pmus = None
    write_filename = "out.txt"
    test_type = "Full"
    noise = 0.0
    test_scenarios = None
    use_filter = True
    verbose = False
    for i in range(1, len(args), 2):
        flag, arg = args[i], args[i+1]
        if flag == "-network":
            filename = arg
        elif flag == "-pmus":
            if arg[0:3] == "All":
                pmus = range(int(arg.split(',')[1]))
            else:
                pmus = [int(s) for s in arg.split(',')]
        elif flag == "-write_file":
            write_filename = arg
        elif flag == "-test_type":
            # Can be "Full" or "Single_Lines"
            test_type = arg
        elif flag == "-noise":
            noise = float(arg)
        elif flag == "-test_scenarios":
            test_scenarios = []
            for strkey in arg.split(';'):
                sub_index, splitting_nodes = strkey.split(',')
                sub_index = int(sub_index)
                splitting_nodes = tuple([int(s) for s in 
                                         splitting_nodes.split(' ')])
                test_scenarios.append((sub_index, splitting_nodes))
        elif flag == "-use_filter":
            if arg == "True":
                use_filter = True
            else:
                use_filter = False
        elif flag == "-verbose":
            verbose = bool(arg)
        else:
            print "Unknown flag: {0}".format(flag)
            assert False
    assert filename is not None
    assert pmus is not None
    return (filename, pmus, write_filename, test_type, 
            noise, test_scenarios, use_filter, verbose)

def main(args):
    """Run a test of FLiER.

    Parameters are passed to the function in pairs, where args[i] is a flag
    and args[i+1] is a parameter value. Valid flag, value combinations are the 
    following:
    -network [filename]: A .cdf or .m file containing the network to be read in.
       A .cdf file is in the IEEE Common Data Format (see 
       https://www.ee.washington.edu/research/pstca/). A .m file is in MATPOWER
       format.
    -use_filter ["True", "False"]: Whether or not to use the FLiER filtering
       procedure. Default is True.
    -pmus [int,...,int]: A list of the indices of buses on which to place PMUs.
    -test_type ["Full", "Single_Lines"]: If "Full", perform a test of a 
       substation splitting into two nodes. If "Single_Lines", perform a line 
       failure test. Default is "Full".
    -test_scenarios [**]: A list of the scenarios to test.
    -noise [double]: Add noise to pre- and post-event voltage readings. Value 
       specifies standard deviation of noise. Default is 0.0.
    -write_file [filename]: Where to write the output data. Default is "out.txt".
    -verbose ["True", "False"]: Turn verbose output to the command-line on or 
       off. Default is "False".

    **test_scenarios format: A semicolon-separated list in which each item is a 
       scenario. Each item is a substation index, a comma, and then a 
       space-separated list of substation nodes that split from the substation.
       For example, "86,1 2 3;176,1 2;539,7;702,4 5".
    """
    print "Running FLiER test with the following options:"
    print "\n".join(["{0}: {1}".format(args[i], args[i+1]) for i in range(1, len(args), 2)])
    out = read_inputs(args)
    (filename, pmus, write_filename, test_type, 
             noise, test_scenarios, use_filter, verbose) = out

    if filename.endswith('.m'):
        powernet = PowerNetwork.powernet_from_matpower(filename)
    else:
        powernet = PowerNetwork.powernet_from_IEEECDF(filename)
    # Simulate the pre-event network voltages.
    V = powernet.simulate()
    pre_event_volts = np.hstack([np.angle(V), np.abs(V)])
    curr_FLiER = FLiER_Substation(powernet, pre_event_volts, pmus, verbose=verbose)
    
    solution_ranks = []
    filter_ranks = []
    num_tests = 0
    num_missed = 0
    file = open(write_filename, 'w')
    # Iterate over test cases. For each case, make the specified change to the
    # network and simulate the resulting voltages. Then run FLiER to attempt to
    # identify the change that occurred.
    for sub in curr_FLiER.substations:
        # For each substation, iterate over possible tests. The set of tests 
        # returned by the iterator depends on test_type.
        for splitting_nodes in sub.node_iterator(test_type):
            key = (sub.index, tuple(splitting_nodes))
            if test_scenarios is not None and key not in test_scenarios:
                continue
            if verbose:
                if test_type == 'Full':
                    print "On bus {0}, splitting nbrs {1}".format(sub.index, 
                                                                  splitting_nodes)
                else:
                    bab = sub.bus_across_branch(splitting_nodes[0])
                    print "On line {0}".format((sub.index, bab.index))
            try:
                # Make change according to test scenario, simulate results.
                # Some changes result in simulation nonconvergence. In this 
                # case, skip the scenario.
                post_event_volts = get_post_event_volts(sub, splitting_nodes, 
                                                        powernet, curr_FLiER)
            except RuntimeError as e:
                print str(e)
                continue
                
            if noise > 0:
                # Add noise if called for.
                noisy_prev = pre_event_volts.copy()
                noisy_prev += np.random.normal(0, noise, len(noisy_prev))
                noisy_postv = post_event_volts.copy()
                noisy_postv += np.random.normal(0, noise, len(noisy_postv))
                noisy_FLiER = FLiER_Substation(powernet, noisy_prev, pmus)
                out = noisy_FLiER.find_topology_error(noisy_postv, test_type,
                                                      use_filter)
            else:               
                # Run FLiER
                out = curr_FLiER.find_topology_error(post_event_volts, test_type,
                                                     use_filter)
            scores, filter_scores, fraction_ts_computed = out
            solkey = (sub.index, tuple(splitting_nodes)) # Solution key
            
            num_tests += 1
            if scores is False:
                continue
            if solkey not in scores:
                if verbose:
                    print "Solution not present"
                num_missed += 1
                score_of_solution = np.Inf
            else:
                score_of_solution = scores[solkey]
                rank = np.sum(scores.values() <= score_of_solution)
                buses_ordered = [y[0][0] if y[0] != -1 else -1 for y in 
                                 sorted(scores.items(), key = lambda x : x[1])]
                # bus_rank is the rank of the correct answer if we only care about
                # getting the substation index correct and not the specific nodes
                # that split.
                bus_rank = buses_ordered.index(sub.index) + 1
            if score_of_solution == np.Inf:
                # Bit of a hack here. Occasionally the filter removes the 
                # correct answer entirely from the list of possibilities. In 
                # this case we just set the rank of the solution to something 
                # large.
                rank = 300
                bus_rank = 300
                
            if verbose:
                print "Solution score: {0}".format(score_of_solution)
                print "Solution rank: {0}".format(rank)
                print "Correct bus rank: {0}".format(bus_rank)
                print "Number scores not filtered: {0}".format(len(scores))
            
            # Update scoring data structures. Write the results of this test to
            # the output file.
            solution_ranks.append(rank)
            sol_filter_score = filter_scores[solkey]
            filter_ranks.append(np.sum(filter_scores.values() <= sol_filter_score))
            write_to_file(file, solkey, out, curr_FLiER.substations)
            
    file.close()
    
    print num_tests
    print "{0} missed entirely".format(num_missed)
    sorted_solranklist, sorted_filtersolranklist = zip(*sorted(zip(solution_ranks, filter_ranks), key = lambda x : x[0]))
    print "Mean solution rank: {0}, Mean filtered solution rank: {1}".format(np.mean(sorted_solranklist), np.mean(sorted_filtersolranklist))
    print "Median sol rank: {0}, median filtered sol rank: {1}".format(np.median(sorted_solranklist), np.median(sorted_filtersolranklist))
    print "Frac correct: {0}, frac filtered correct: {1}".format(np.sum(np.array(sorted_solranklist) == 1) / float(num_tests), np.sum(np.array(sorted_filtersolranklist) == 1) / float(num_tests))
    print "Frac in top 3: {0}, frac filtered in top 3: {1}".format(np.sum(np.array(sorted_solranklist) <= 3) / float(num_tests), np.sum(np.array(sorted_filtersolranklist) <= 3) / float(num_tests))
    
    
if __name__ == '__main__':
    """Run a test of FLiER, either of substation splitting or of line failure.

    Some command-line examples:
    python FLiER_Substation_Test.py -network ../networks/ieee57cdf.txt -pmus 3,12,33 -test_type Full -write_file ieee57subs_4_13_34.txt

    python FLiER_Substation_Test.py -network ../networks/ieee57cdf.txt -pmus 3,12,33 test_type Single_Lines -write_file ieee57lines_4_13_34.txt
    """
    main(sys.argv)
