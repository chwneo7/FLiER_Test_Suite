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
from FLiER import FLiER
from DC_FLiER import DC_FLiER
import sys
import numpy as np
import pylab

def single_FLiER_test(power_network, pmus, noise, branch_to_fail, use_filter=True,
                      pre_event_volts = None, FLiER_type = "Jacobian", 
                      verbose=False):
    """
    Run a FLiER test for a specific line failure.
    
    Inputs:
    power_network - The power network in which the test will occur.
    pmus - A list of bus indices with PMUs.
    line_to_fail - The transmission line that will fail.
    
    Outputs:
    score_dict - A dictionary with keys that are indices into the 'edges' list
       and values that are tij's. The "no line failure" case has
       key -1.
    ts_computed - The number of tij's that were computed (lines that got past 
       the filter).
    """
    powernet = power_network.copy()
    Y = powernet.Y.copy()
    
    try:
        if pre_event_volts is None:
            V = powernet.simulate()
            pre_event_volts = np.hstack([np.angle(V), np.abs(V)])
                
        powernet.remove_branch(branch_to_fail)
    
        V = powernet.simulate()
        post_event_volts = np.hstack([np.angle(V), np.abs(V)])
        
    except RuntimeError as e:
        # If the solver for the power flow equations fails to converge.
        if verbose:
            print str(e)
        return False, False, False
        
    if noise > 0:
        post_event_volts += np.random.normal(0, noise, len(post_event_volts))
    
    # Get the matrix that projects a voltage vector onto the subspace
    # rendered observable by the PMUs.
    E = PowerNetwork.get_observable_projection_matrix(Y, pmus)
    
    # Run FLiER
    if FLiER_type == "Jacobian":
        res = FLiER(Y, powernet.branches, pre_event_volts, post_event_volts, 
                    E, powernet.buses, powernet.slack_inds, 
                    use_filter=use_filter, verbose=verbose)
    elif FLiER_type == "DC_Approximation":
        res = DC_FLiER(powernet.dc_mat, powernet.branches, pre_event_volts, 
                       post_event_volts, E, powernet.slack_inds, verbose=verbose)
    else:
        assert False
    score_dict, initial_score_dict, fraction_ts_computed = res

    return score_dict, initial_score_dict, fraction_ts_computed
    
def network_FLiER_test(network_filename, pmus, noise, use_filter=True, 
                       test_branches = None, FLiER_type = "Jacobian", 
                       verbose=False):
    """
    Simulate the failure of each line in the network one at a run, and run
    FLiER for each case.
    
    Inputs:
    network_filename - The path to the file containing the network in IEEE CDF
       format.
    pmus - A list of bus indices with PMUs on them.
    
    Outputs:
    
    """
    if network_filename.endswith('.m'):
        powernet = PowerNetwork.powernet_from_matpower(network_filename)
    else:
        powernet = PowerNetwork.powernet_from_IEEECDF(network_filename)
    num_lines = len(powernet.branches)
    solution_ranks = [] # Contains ranks of correct answers
    score_list = [] # Contains tij's
    fraction_ts_computed_list = [] # How many tij's computed per test.
    dict_of_score_dicts = dict()
    num_filtered_out = 0
    
    V = powernet.simulate()
    pre_event_volts = np.hstack([np.angle(V), np.abs(V)])

    if test_branches is None:
        test_branches = range(len(powernet.branches))
    for i in test_branches:
        branch = powernet.branches[i]
        if verbose:
            print "On line {0}".format([bus.index for bus in branch.buses])
        out = single_FLiER_test(powernet, pmus, noise, branch_to_fail=branch, 
                                use_filter=use_filter, pre_event_volts=pre_event_volts,
                                FLiER_type=FLiER_type, verbose=verbose)
        out = score_dict, initial_score_dict, fraction_ts_computed
        if score_dict == False:
            # then power flow equation solver did not converge
            continue
        
        fraction_ts_computed_list.append(fraction_ts_computed)
        if i in score_dict:
            score_of_solution = score_dict[i]
            rank = np.sum(score_dict.values() <= score_of_solution)
            solution_ranks.append(rank)
        else:
            # Then the correct line was filtered out.
            # Just choose the rank as last, and some large tij
            # filler value.
            if verbose:
                print "Solution filtered out."
            score_of_solution = 100
            rank = num_lines
            solution_ranks.append(rank)
            num_filtered_out += 1
        dict_of_score_dicts[i] = (score_dict, initial_score_dict, fraction_ts_computed)
            
        # Produce a list of (key, value) dictionary elements sorted by value.
        # Then split that list to create a sorted list of keys and a sorted list of
        # values. Recall that the keys are indices into the list of edges.
        branch_inds, scores = zip(*sorted(score_dict.iteritems(), key=lambda x: x[1]))
        
        if verbose:
            print score_of_solution, rank, len(score_dict)
        score_list.append((scores, score_of_solution))
        
    if verbose:
        print "{0} filtered out entirely.".format(num_filtered_out)
    return score_list, dict_of_score_dicts, powernet.branches, solution_ranks, fraction_ts_computed_list
    
def write_to_file(write_filename, score_dict_of_dicts, branches):
    f = open(write_filename, 'w')
    for key, val in score_dict_of_dicts.iteritems():
        buses = [bus.index for bus in branches[key].buses]
        f.write("{0}; {1}, {2}\n".format(str(key), str(buses[0]), str(buses[1])))
        score_dict, filter_score_dict, fraction_ts_computed = val
        f.write("{0}\n".format(str(fraction_ts_computed)))
        for key, val in score_dict.iteritems():
            if key == -1:
                f.write("{0}; {1}\n".format(str(key), str(val)))
            else:
                buses = [bus.index for bus in branches[key].buses]
                f.write("{0}; {1}, {2}; {3}\n".format(str(key), str(buses[0]),
                                                      str(buses[1]), str(val)))
        f.write("\n")
        for key, val in filter_score_dict.iteritems():
            if key == -1:
                f.write("{0}; {1}\n".format(str(key), str(val)))
            else:
                buses = [bus.index for bus in branches[key].buses]
                f.write("{0}; {1}, {2}; {3}\n".format(str(key), str(buses[0]),
                                                      str(buses[1]), str(val)))
        f.write("END TEST\n\n")
    f.close()
    
def read_inputs(args):
    filename = None
    pmus = None
    write_filename = "out.txt"
    noise = 0.0
    test_scenarios = None
    use_filter = True
    FLiER_type = "Jacobian"
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
        elif flag == "-noise":
            noise = float(arg)
        elif flag == "-test_scenarios":
            test_scenarios = [int(s) for s in arg.split(',')]
        elif flag == "-use_filter":
            if arg == "True":
                use_filter = True
            else:
                use_filter = False
        elif flag == "-FLiER_type":
            FLiER_type = arg
        elif flag == "-verbose":
            verbose = bool(arg)
        else:
            print "Unknown flag: {0}".format(flag)
            assert False
    assert filename is not None
    assert pmus is not None
    return (filename, pmus, write_filename, noise, test_scenarios, use_filter, 
            FLiER_type, verbose)

def main(args):
    print "Running FLiER test with the following options:"
    print "\n".join(["{0}: {1}".format(args[i], args[i+1]) for i 
                     in range(1, len(args), 2)])
    (filename, pmus, write_filename, noise, 
              test_branches, use_filter, FLiER_type, verbose) = read_inputs(args)
        
    # Run test
    out  = network_FLiER_test(filename, pmus, noise, use_filter, 
                              test_branches, FLiER_type, verbose)
    (solution_score_list, dict_of_score_dicts, branches, 
            solution_rank_list, fraction_ts_computed_list) = out
    
    write_to_file(write_filename, dict_of_score_dicts, branches)
    
    total = len(solution_rank_list) # number of tests
    
    solution_ranks = np.array(solution_rank_list)
    print "\nFraction correct: {0}".format(np.sum(solution_ranks == 1) / float(total))
    print "Fraction in top 3: {0}".format(np.sum(solution_ranks <= 3) / float(total))
    print "Mean rank: {0}".format(np.mean(solution_ranks))
    print "Stddev of rank: {0}".format(np.std(solution_ranks))
    print "\n"
        
if __name__ == '__main__':
    main(sys.argv)
