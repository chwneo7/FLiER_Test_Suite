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

# FLiER_Test_Suite

This repository contains code to solve power flow equations, as well as to run
the FLiER algorithms discussed in our paper.

Some example python calls, to be run in the /Code directory, are the following:


python FLiER_Test.py -network ../Networks/ieee57cdf.txt -pmus 3,12,33 -use_filter True -noise 0.0 -write_file output.txt

python FLiER_Substation_Test.py -network ../Networks/ieee57cdf.txt -pmus 3,12,33 -use_filter True -noise 0.0 -test_type Full -write_file output.txt

python FLiER_Substation_Test.py -network ../Networks/ieee57cdf.txt -pmus 3,12,33 -use_filter True -noise 0.0 -test_type Single_Lines -write_file output.txt


Note that our code 0-indexes network buses, while e.g. MATLAB 1-indexes buses. So, for example, placing PMUs on 0-indexed buses 3, 12, and 33 means placing PMUs on 1-indexed buses 4, 13, and 34.


# Summary of Files

**Code/Bus.py.** This file contains Bus and Substation objects for a network.

**Code/Branch.py** This file contains the Branch object.

**Code/DC_FLiER.py** This file contains the FLiER algorithm for branch failures using
  the DC load flow matrix instead of the Jacobian matrix to identify voltage
  fingerprints.

**Code/FLiER.py.** This file contains the FLiER algorithm specifically for branch
failures. It runs FLiER by applying the Sherman-Morrison-Woodbury formula.

**Code/FLiER_Substation.py.** This file contains the FLiER algorithm for substation
splittings. Note that branch failures can be approximately simulated as a
substation split by splitting a substation so that the branch in question leads
to a zero-injection leaf node.

**Code/FLiER_Substation_Test.py** This file contains the script to run tests
  using FLiER_Substation.py.

**Code/FLiER_Test.py** This file contains the script to run tests
  using FLiER.py.

**Code/PowerNetwork.py** This file contains the PowerNetwork object.

**Code/Power_Utils.py** This file contains some useful functions related to
  power networks.

**Code/run_tests.py** This file contains a script to run a series of tests using
  both FLiER_Test.py and FLiER_Substation_Test.py. Many of the tests are
  commented out. Uncomment the tests you wish to run.

**Code/Utils.py** This file contains some useful functions not necessarily
  related to power networks.



