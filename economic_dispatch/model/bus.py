# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for Bus.
"""

import pyomo.environ as pyo

from economic_dispatch.model.units.base import _Unit
from economic_dispatch.model.factory import UnitFactory


def _get_index(name, from_list):
    for index, u in enumerate(from_list):
        if u.name == name:
            return index
    return None


class Bus:
    """
    Bus containing all units related to this bus

    ...

    Attributes
    ----------
    name: str
        name of the bus
    units: dict
        key: unit name, value: unit object
    unit_names: list
        list of unit names

    Methods
    -------
    block_rule(block: pyo.Block)
        Builds all the optimisation model components for this unit type 
        on top of the passed block.
    topics(prefix: bool=True)
        returns a list of network's units' problem instance parameter names, 
        if prefix=True '{unit.name}.' is added before attribute names.
    add(unit: _Unit)
        add unit to bus
    remove(name: str)
        remove unit from bus
    """

    def __init__(self, name, units=None):
        """
        Parameters
        ----------
        name: str
            name of the bus
        units: List
            list of all unit objects related to this bus
        """
        if units is None:
            units = []

        self.name = name
        self._units = units

    def __repr__(self):
        return self.__class__.__name__ + '(units=' + str(list(self.unit_names)) + ')'

    def add(self, unit):
        """ Adds unit to bus. """
        if isinstance(unit, _Unit):
            self._units.append(unit)
        else:
            raise ValueError("Input unit is not valid")

    def remove(self, name):
        """ Removes unit from bus. """
        if name in self.unit_names:
            self._units.pop(_get_index(name, self._units))

    def topics(self, prefix=True):
        """ Return a list of bus's units' problem instance parameter names.

        If prefix=True '{unit.name}.' is added before attribute names.
        """
        return [t for u in self._units for t in u.topics(prefix=prefix)]

    @property
    def units(self):
        """ Dictionary with unit names as keys and unit objects as values. """
        return {u.name: u for u in self._units}

    @property
    def unit_names(self):
        """ List of unit names. """
        return list(self.units.keys())

    def clear(self):
        """ Sets problem instance parameter values of units to None. """
        for unit in self._units:
            unit.clear()

    def ready(self):
        """ Returns True if all problem instance parameter values are ready for all units. """
        return all(unit.ready() for unit in self._units)

    def block_rule(self, block):
        """
        Builds all the optimisation model components for this bus
        on top of the passed block.

        The underlying model should have its time index set at attr T

        Parameters
        ----------
        block: pyo.Block
            bus level block of the model.
        """

        model = block.model()
        network = block.parent_block()

        def U_init(b):
            return self.unit_names

        block.U = pyo.Set(initialize=U_init)

        # Block rules to units
        def unit_rule(b, i):
            self.units[i].block_rule(b)

        block.Units = pyo.Block(block.U, rule=unit_rule)

        # Optimized units
        def controllable_init(b):
            return [u for u in b.U if hasattr(b.Units[u], 'dispatch')]

        block.C = pyo.Set(initialize=controllable_init)

        # Total real power injection to bus
        def inj_rule(b, i):
            return sum(network.icd_matrix[self.name, line] * network.Lines[line].power_flow[i]
                       for line in network.L)

        block.net_injection = pyo.Expression(model.T, rule=inj_rule)

        # Dispatches minus electrical loads
        def rp_rule(b, i):
            return sum(b.Units[u].real_power[i] for u in b.U)

        block.real_power = pyo.Expression(model.T, rule=rp_rule)

        # Power balance constraint
        def demand_rule(b, i):
            return b.real_power[i] == b.net_injection[i]

        block.demand_balance = pyo.Constraint(model.T, rule=demand_rule)

        # Cost
        def op_cost_rule(b, i):
            return sum(b.Units[u].operational_cost[i] for u in b.U)

        block.operational_cost = pyo.Expression(model.T, rule=op_cost_rule)

        # Cost
        def cost_rule(b, i):
            return sum(b.Units[u].cost[i] for u in b.U)

        block.cost = pyo.Expression(model.T, rule=cost_rule)

    @classmethod
    def from_json(cls, json_bus):
        """ Creates a Bus from dictionary and returns it. """
        json_units = json_bus.get("units")
        units = []
        for json_unit in json_units:
            unit = UnitFactory.make_component(**json_unit)
            units.append(unit)
        
        json_bus["units"] = units
        bus = Bus(**json_bus)
        return bus