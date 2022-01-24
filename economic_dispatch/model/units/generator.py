# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for unit level component StorageUnit.
"""


import pyomo.environ as pyo

from economic_dispatch.model.units.base import _Unit


class Generator(_Unit):
    """
    A Generator unit

    Attributes
    ----------
    name: str
        name of the unit
    power_max: float
        maximum power
    power_min: float
        minimum power
    ramp_limits: Tuple[float, float]
        ramp limts down and up

    Methods
    -------
    block_rule(block: pyo.Block)
        Builds all the optimisation model components for this unit type
        on top of the passed block.
    topics(prefix: bool=True)
        returns a list of unit's problem instance parameter names,
        if prefix=True '{self.name}.' is added before attribute names.
    """
    
    PROB_INSTANCE_PARAMS = []
    TIME_INDEXED_PROB_INSTANCE_PARAMS = []

    def __init__(self, name, power_max=None, power_min=0.0, ramp_limits=None):
        """
        Parameters
        ----------
        name: str
            name of the unit
        power_max: float
            maximum power
        power_min: float
            minimum power
        ramp_limits: Tuple[float, float] or float
            ramp rate limits down and up
        """

        super().__init__(name)
        
        self.power_max = power_max
        self.power_min = power_min
        self.ramp_limits = ramp_limits
    
    @property
    def ramp_limits(self):
        """ Tuple of ramp rate limits, down and up. """
        return self._ramp_limits

    @ramp_limits.setter
    def ramp_limits(self, val):
        """ Sets value to 2-tuple from tuple, single value or None. """
        if val is None:
            # None means ramp bound is max possible change
            diff = self.power_max-self.power_min
            self._ramp_limits = (diff, diff)        
        elif isinstance(val, tuple):
            self._ramp_limits = val
        else:
            # singel value -> same down and up
            self._ramp_limits = (val, val)
    
    def block_rule(self, block):
        """
        Builds all the optimisation model components for this unit type
        on top of the passed block.

        The underlying model should have its time index set at attr T

        Parameters
        ----------
        block: pyo.Block
            unit level block of the model.
        """

        model = block.model()

        def bounds_rule(b, i):
            return self.power_min, self.power_max
        block.gen = pyo.Var(model.T, bounds=bounds_rule)

        # Constrain downward ramp rate
        def ramp_down_rule(b, i):
            if i == model.T.first():
                pyo.Constraint.Skip
            return b.gen[i] - b.gen[i-1] >= -self.ramp_limits[0]
        block.ramp_down_constr = pyo.Constraint(model.T, rule=ramp_down_rule)

        # Constrain upward ramp rate
        def ramp_up_rule(b, i):
            if model.T.first():
                pyo.Constraint.Skip
            return b.gen[i] - b.gen[i-1] <= self.ramp_limits[1]
        block.ramp_up_constr = pyo.Constraint(model.T, rule=ramp_up_rule)

        def rp_rule(b, i):
            return b.gen[i]
        block.real_power = pyo.Expression(model.T, rule=rp_rule)

        def dispatch_rule(b, i):
            return b.gen[i]
        block.dispatch = pyo.Expression(model.T, rule=dispatch_rule)

        def operational_cost_rule(b, i):
            return 0.0
        block.operational_cost = pyo.Expression(model.T, rule=operational_cost_rule)

        def cost_rule(b, i):
            return b.operational_cost[i]
        block.cost = pyo.Expression(model.T, rule=cost_rule)
