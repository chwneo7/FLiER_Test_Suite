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

class Branch:
    """A branch represents a transmission line in a power network.

    More generally, it can represent any connection between two network buses
    that involve non-infinite admittance.
    """
    def __init__(self, ID, buses, impedance, line_charging_susc = 0.0,
                 nonunity_tap = False, tap = 1.0+0.0j):
        self.ID = ID
        self.buses = buses
        self.impedance = impedance
        self.admittance = 1.0 / impedance
        self.line_charging_susc = line_charging_susc
        self.nonunity_tap = nonunity_tap
        self.tap = tap
        
        self.buses[0].add_branch(self)
        self.buses[1].add_branch(self)
        
    def copy(self):
        return Branch(self.ID, self.buses, self.impedance,
                      self.line_charging_susc, self.nonunity_tap, self.tap)
