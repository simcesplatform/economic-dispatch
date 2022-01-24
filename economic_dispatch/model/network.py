# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for Network.
"""

import pyomo.environ as pyo

from economic_dispatch.model.bus import Bus
from economic_dispatch.model.transmission import Line
from economic_dispatch.model.units import Retailer
from economic_dispatch.model.units import StorageUnit


def _get_index(name, from_list):
    for index, v in enumerate(from_list):
        if v.name == name:
            return index
    return None


class Network:
    """
    Network containing buses and lines

    ...

    Attributes
    ----------
    name: str
        name of the network
    buses: dict
        key: bus name, value: bus object
    lines: dict
        key: line name, value: line object
    units: dict
        key: unit name, value: unit object

    Methods
    -------
    block_rule(block: pyo.Block)
        Builds all the optimisation model components for this unit type 
        on top of the passed block.
    topics(prefix: bool=True)
        returns a list of network's units' problem instance parameter names, 
        if prefix=True '{unit.name}.' is added before attribute names.
    add_bus(bus: Bus)
        add bus to network
    remove_bus(name: str)
        remove bus form network
    add_line(line: Line)
        add line to network
    remove_line(name: str)
        remove line form network
    """

    def __init__(self, buses=None, lines=None, name="Network"):
        """
        Parameters
        ----------
        name: str
            name of the network
        buses: List
            list of the bus objects
        lines:
            list of the line objects
        """

        if lines is None:
            lines = []
        if buses is None:
            buses = []

        self._buses = []
        for bus in buses:
            self.add_bus(bus)

        self._lines = []
        for line in lines:
            self.add_line(line)

        self.name = name

    def add_bus(self, bus):
        """ Adds a bus to network. """
        if isinstance(bus, dict):
            bus = Bus.from_json(bus)

        if isinstance(bus, Bus):
            self._buses.append(bus)
        else:
            raise ValueError

    def remove_bus(self, name):
        """ Removes a bus from network. """
        if name in self.bus_names:
            self._buses.pop(_get_index(name, self._buses))
    
    def add_line(self, line):
        """ Adds a line to network. """
        if isinstance(line, dict):
            line = Line.from_json(line)

        if isinstance(line, Line):
            self._lines.append(line)
        else:
            raise ValueError

    def remove_line(self, name):
        """ Removes a line from network. """
        if name in self.line_names:
            self._lines.pop(_get_index(name, self._lines))

    def topics(self, prefix=True):
        """ Return a list of network's units' problem instance parameter names.

        If prefix=True '{unit.name}.' is added before attribute names.
        """
        return [t for b in self._buses for t in b.topics(prefix=prefix)]

    @property
    def buses(self):
        """ Dictionary with bus names as keys and bus objects as values. """
        return {b.name: b for b in self._buses}

    @property
    def bus_names(self):
        """ List of bus names. """
        return list(self.buses.keys())

    @property
    def lines(self):
        """ Dictionary with line names as keys and line objects as values. """
        return {l.name: l for l in self._lines}

    @property
    def line_names(self):
        """ List of line names. """
        return list(self.lines.keys())

    @property
    def units(self):
        """ Dictionary with unit names as keys and unit objects as values. """
        return {u.name: u for b in self._buses for u in b.units.values()}

    @property
    def unit_names(self):
        """ List of unit names. """
        return list(self.units.keys())

    @property
    def retailer_names(self):
        """ List of Retailer unit names. """
        return [name for name, u in self.units.items() if isinstance(u, Retailer)]

    @property
    def storage_names(self):
        """ List of Storage unit names. """
        return [name for name, u in self.units.items() if isinstance(u, StorageUnit)]

    def clear(self):
        """ Sets problem instance parameter values of units to None. """
        for bus in self._buses:
            bus.clear()

    def ready(self):
        """ Returns True if all problem instance parameter values are ready for all units. """
        return all(bus.ready() for bus in self._buses)

    def block_rule(self, block):
        """
        Builds all the optimisation model components for the network
        on top of the passed block.

        The underlying model should have its time index set at attr T

        Parameters
        ----------
        block: pyo.Block
            network level block of the model.
        """

        model = block.model()

        def B_init(b):
            return self.bus_names
        block.B = pyo.Set(initialize=B_init)

        def L_init(b):
            return self.line_names
        block.L = pyo.Set(initialize=L_init)

        def bus_rule(b, i):
            self.buses[i].block_rule(b)
        block.Buses = pyo.Block(block.B, rule=bus_rule)

        def line_rule(b, i):
            self.lines[i].block_rule(b)
        block.Lines = pyo.Block(block.L, rule=line_rule)

        def op_cost_rule(b, i):
            return sum(b.Buses[bus].operational_cost[i] for bus in b.B)
        block.op_cost = pyo.Expression(model.T, rule=op_cost_rule)

        def cost_rule(b, i):
            return sum(b.Buses[bus].cost[i] for bus in b.B)
        block.cost = pyo.Expression(model.T, rule=cost_rule)
        
    @classmethod
    def from_json(cls, json_network):
        """ Creates a  Network from dictionary and returns it. """
        json_buses = json_network.get("buses")
        if json_buses is None:
            json_buses = [{"name": "single-point", "units": json_network["units"]}]
            del json_network["units"]
        buses = []
        for json_bus in json_buses:
            bus = Bus.from_json(json_bus)
            buses.append(bus)
        json_network["buses"] = buses

        json_lines = json_network.get("lines")
        lines = []
        if json_lines is not None:
            for json_line in json_lines:
                line = Line(**json_line)
                lines.append(line)
        json_network["lines"] = lines

        network = Network(**json_network)
        return network
