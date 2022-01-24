# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains classes for EconomicDispatchPlanner.
"""

from typing import List, Tuple, Dict, Union, Any

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from economic_dispatch.utils import Barrier
from economic_dispatch.planner.exceptions import ModelBuildError
from economic_dispatch.model.network import Network


SOLVED_STATUS = "Solved"
UNSOLVED_STATUS = "Unsolved"


class EconomicDispatchPlanner:
    """
    Planner

    Attributes
    ----------
    name: str
        name of the planner
    horizon: int
        num of timesteps in model
    time_index: List[int]
        range from 0 to horizon-1
    timestep: float
        timestep length in hours
    network: Network
    model: pyo.ConcreteModel
        pyomo optimisation model object
        T: time index set
        timestep: timestep in h
        Network: Network Block
        OBJ: objective
    solve_status: str

    Methods
    -------
    set_param(resource_id, param_name, val)
        sets parameter value for resource
    ready_to_solve() -> bool
        is model ready to be built and solved
    solve()
        solve model
    results()
        returns results in dict for solved model
    clear()
        clears barrier, parameters need to be set again
    topics(prefix: bool=True)
        returns a list of model's network's units' problem instance parameter names, 
        if prefix=True '{unit.name}.' is added before attribute names.

    Class Methods
    -------------
    from_json(ed_json: dict)
    """

    def __init__(self, network, horizon=36, name='Planner', timestep=1.0, solver='glpk'):
        """
        Parameters
        ----------
        network: Network object or dict description of network
            network containing the grid composition and units
        horizon: int
            optimization horizon length (default=36)
        name: str
            name of planner (default='Planner')
        timestep: float
            optimisation timestep
        solver: str
            name of solver (default='glpk')
        """

        self.name = name
        self.horizon = horizon

        if isinstance(network, dict):
            network = Network.from_json(network)
        self.network = network
        self.timestep = timestep

        self.model = None
        self._solver = SolverFactory(solver)
        self._pyo_results = None

        self._shallow_clear = True
        self._barrier = Barrier(self.topics())

    def _get_status(self, results):
        """ Takes results returned by pyomo solve and return status. """
        if results is None:
            return UNSOLVED_STATUS
        elif results.solver.status == "ok" and results.solver.termination_condition != "infeasible":
            return SOLVED_STATUS
        return UNSOLVED_STATUS # another status

    def topics(self, prefix=True):
        """ Return a list of model's network's units' problem instance parameter names.

        If prefix=True '{unit.name}.' is added before attribute names.
        """
        return self.network.topics(prefix=prefix)

    @property
    def units(self):
        """ Dictionary with unit names as keys and unit objects as values. """
        return self.network.units

    @property
    def unit_names(self):
        """ List of unit names. """
        return self.network.unit_names

    @property
    def retailer_names(self):
        """ List of Retailer unit names. """
        return self.network.retailer_names
    
    @property
    def solve_status(self):
        return self._get_status(self._pyo_results)

    @property
    def time_index(self):
        """ Time index list from 0 to horizon-1. """
        return [*range(0, int(self.horizon), 1)]

    def set_param(self, resource_id, param_name, val):
        """ Sets attribute param_name of a resource named resource_id to value. """
        if resource_id not in self.units.keys():
            raise KeyError("No unit with resource id {:s}".format(resource_id))

        if param_name in self.units[resource_id].time_indexed_parameters() and not self._check_time_indexed_value(val):
            raise ValueError("Parameter is time indexed, length or type of value does not match.")

        self.units[resource_id].__setattr__(param_name, val)
        self._barrier.process_arrive(resource_id+'.'+param_name)

    def _check_time_indexed_value(self, val: list):
        return isinstance(val, list) and len(val) == self.horizon

    def ready_to_solve(self):
        """ Returns True if model is ready to solve. """
        if self._shallow_clear:
            return self._barrier.pass_state
        return self.network.ready()

    def build_model(self):
        """ Builds pyomo model object. """
        model = pyo.ConcreteModel()

        model.timestep = pyo.Param(initialize=self.timestep)
        model.T = pyo.Set(initialize=self.time_index)
        model.Network = pyo.Block(rule=self.network.block_rule)

        def price_rule(m):
            return sum(m.Network.op_cost[i] for i in model.T)
        model.price = pyo.Expression(rule=price_rule)
        
        def objective(m):
            return sum(m.Network.cost[i] for i in model.T)
        model.OBJ = pyo.Objective(rule=objective)

        self.model = model
        self._pyo_results = None

    def _get_results(self):
        """ Retrieves and returns dispatch results in a dictionary. """
        results = {}

        for bus_name in self.model.Network.B:
            bus = self.model.Network.Buses[bus_name]
            for unit_name in bus.C:
                unit = bus.Units[unit_name]
                values = [pyo.value(unit.dispatch[i]) for i in self.model.T]
                results[unit_name] = values

        return results

    def results(self):
        """ Return results in a dictionary. If model is not solved returns None. """
        if self.solve_status == SOLVED_STATUS:
            return self._get_results()
        else:
            return None

    def clear(self):
        """
        Resets barrier so that every parameter needs to be set again 
        before economic dispatch is ready to solve.

        if shallow is False all parameter values are cleared.
        """
        self._pyo_results = None
        self._barrier.reset()

        if not self._shallow_clear:
            self.network.clear()

    def solve(self):
        """ Solves the model if model is ready. """

        results = None
        try:
            self.build_model()
        except:
            raise ModelBuildError("Unable to build model from data.")
        else:
            results = self._solver.solve(self.model)

        self._pyo_results = results

        return results

    @classmethod
    def from_json(cls, ed_json: Dict):
        """ Creates an EconomicDispatchPlanner from dictionary and returns it. """
        ed = EconomicDispatchPlanner(**ed_json)
        return ed