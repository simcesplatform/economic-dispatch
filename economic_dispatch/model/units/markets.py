# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for unit level component Retailer.
"""

import pyomo.environ as pyo
from economic_dispatch.model.units.base import _Unit

# Multiplies price of diversion from commitment
DIVERSION_MULTIPLIER_UP = 2
DIVERSION_MULTIPLIER_DOWN = -0.5
# Up multiplier 1 = raising consumption raises payed price
# Down multiplier 0.5 = lowering consumption lowers payed price at half value


class Retailer(_Unit):
    """
    Electricity retailer unit

    ...

    Attributes
    ----------
    name: str
        name of the unit
    prices: List[float]
        list of prices
    commitments: Dict[int, float]
        timing and commitment level value
    max_demand: float
        maximum demand, default=None (infinity)
    diversion_cost: float
        penalty for diverging from commmitment levels
    max_diversion: float
        max diversion from commitments

    Methods
    -------
    block_rule(block: pyo.Block)
        Builds all the optimisation model components for this unit type 
        on top of the passed block.
    topics(prefix: bool=True)
        returns a list of unit's problem instance parameter names, 
        if prefix=True '{self.name}.' is added before attribute names.
    """

    PROB_INSTANCE_PARAMS = ['prices']
    TIME_INDEXED_PROB_INSTANCE_PARAMS = ['prices']

    def __init__(self, name, max_demand=None):
        """
        Parameters
        ----------
        name: str
            name of the unit
        max_demand: float
            maximum demand, default=None (infinity)
        """

        super().__init__(name)

        self.prices = []
        self.commitments = {}

        self.max_demand = max_demand
        self.diversion_cost_up = None
        self.diversion_cost_down = None
        self.max_diversion = None  # Max diversion from commitments / min or max demand

    @property
    def prices(self):
        return self._prices

    @prices.setter
    def prices(self, val):
        if isinstance(val, list):
            self._prices = val
        else:
            raise ValueError("Price forecast should be a list.")

    @property
    def commitments(self):
        return self._commitments

    @commitments.setter
    def commitments(self, val):
        if isinstance(val, dict):
            self._commitments = val
        else:
            raise ValueError("Commitments should be a dict with int keys.")

    def block_rule(self, block):
        """
        Builds all the optimisation model components for this unit type
        on top of the passed block.

        The underlying model should have its time index set at attr T

        Electricity retailer unit -> power optimized at given price
        Electricity retailer unit -> power produced at predefined commitment level for given time steps

        Parameters
        ----------
        block: pyo.Block
            unit level block of the model.
        """

        model = block.model()

        def gen_bounds(b, i):
            if i in self.commitments.keys() and self.commitments[i] is not None:
                return self.commitments[i], self.commitments[i]
            return 0.0, self.max_demand
        block.gen = pyo.Var(model.T, bounds=gen_bounds)

        def prices_rule(b, i):
            return self.prices[i]
        block.prices = pyo.Param(model.T, initialize=prices_rule, within=pyo.Reals, mutable=True)

        if self.max_diversion is not None and self.max_diversion <= 0.0:
            def rp_rule(b, i):
                return b.gen[i]
            block.real_power = pyo.Expression(model.T, rule=rp_rule)

            def dispatch_rule(b, i):
                return b.gen[i]
            block.dispatch = pyo.Expression(model.T, rule=dispatch_rule)

            def cost_rule(b, i):
                return b.prices[i] * b.gen[i] * model.timestep
            block.cost = pyo.Expression(model.T, rule=cost_rule)

            def operational_cost_rule(b, i):
                return b.cost[i]

            block.operational_cost = pyo.Expression(model.T, rule=operational_cost_rule)

        else:
            # Diversion up
            def d_bounds(b, i):
                if i in self.commitments.keys() and self.commitments[i] is not None:
                    return 0.0, self.max_diversion
                else:
                    return 0.0, 0.0
            block.d = pyo.Var(model.T, bounds=d_bounds)

            # Diversion down
            def d2_bounds(b, i):
                if i in self.commitments.keys() and self.commitments[i] is not None:
                    if abs(self.commitments[i]) < 10e-9:
                        commitment = 0.0
                    else:
                        commitment = self.commitments[i]
                    if self.max_diversion is None:
                        return -1*commitment, 0.0
                    else:
                        return -1*min(commitment, self.max_diversion), 0.0
                else:
                    return 0.0, 0.0
            block.d2 = pyo.Var(model.T, bounds=d2_bounds)

            def rp_rule(b, i):
                return b.gen[i] + b.d[i] + b.d2[i]
            block.real_power = pyo.Expression(model.T, rule=rp_rule)

            def dispatch_rule(b, i):
                return b.real_power[i]
            block.dispatch = pyo.Expression(model.T, rule=dispatch_rule)

            def diversion_cost_rule_up(b, i):
                if self.diversion_cost_up is None:
                    return DIVERSION_MULTIPLIER_UP*self.prices[i]
                return self.diversion_cost_up
            block.diversion_cost_up = pyo.Param(model.T, initialize=diversion_cost_rule_up)

            def diversion_cost_rule_down(b, i):
                if self.diversion_cost_down is None:
                    return DIVERSION_MULTIPLIER_DOWN*self.prices[i]
                return self.diversion_cost_down
            block.diversion_cost_down = pyo.Param(model.T, initialize=diversion_cost_rule_down)

            """
            1: Cost of real power -> bought from market
            2: Cost of deviation up -> bought from market (price times diversion multiplier)
            3: Cost of deviation down -> buying less from market (price times (negative) multiplier)
                i.e. buying incurs cost
                and
                deviation in either direction incurs a positive cost
            """
            def cost_rule(b, i):
                return b.prices[i] * b.real_power[i] * model.timestep \
                       + b.diversion_cost_up[i] * b.prices[i] * b.d[i] * model.timestep \
                       + b.diversion_cost_down[i] * b.prices[i] * b.d2[i] * model.timestep
            block.cost = pyo.Expression(model.T, rule=cost_rule)

            def operational_cost_rule(b, i):
                return b.cost[i]
            block.operational_cost = pyo.Expression(model.T, rule=operational_cost_rule)
