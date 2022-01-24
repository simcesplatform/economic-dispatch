# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains classes for unit level components StaticUnit.
"""

import pyomo.environ as pyo
from economic_dispatch.model.units.base import _Unit


class StaticUnit(_Unit):
    """
    Static forecast unit

    ...

    Attributes
    ----------
    name: str
        name of the unit
    forecast: List[float]
        power forecast

    Methods
    -------
    block_rule(block: pyo.Block)
        Builds all the optimisation model components for this unit type 
        on top of the passed block.
    topics(prefix: bool=True)
        returns a list of unit's problem instance parameter names, 
        if prefix=True '{self.name}.' is added before attribute names.
    """

    PROB_INSTANCE_PARAMS = ['forecast']
    TIME_INDEXED_PROB_INSTANCE_PARAMS = ['forecast']

    def __init__(self, name):
        """
        Parameters
        ----------
        name: str
            name of the unit
        """

        super().__init__(name)
        self.forecast = []

    @property
    def forecast(self):
        return self._forecast

    @forecast.setter
    def forecast(self, val):
        if isinstance(val, list):
            self._forecast = val
        else:
            raise ValueError("Forecast should be a list.")

    def block_rule(self, block):
        """
        Builds all the optimisation model components for this unit type
        on top of the passed block.

        The underlying model should have its time index set at attr T

        Generator unit -> power produced given the predefined power forecast

        Parameters
        ----------
        block: pyo.Block
            unit level block of the model.
        """

        model = block.model()

        def gen_rule(b, i):
            return self.forecast[i]
        block.gen = pyo.Param(model.T, initialize=gen_rule, within=pyo.Reals, mutable=True)

        def rp_rule(b, i):
            return b.gen[i]
        block.real_power = pyo.Expression(model.T, rule=rp_rule)

        def operational_cost_rule(b, i):
            return 0.0

        block.operational_cost = pyo.Expression(model.T, rule=operational_cost_rule)

        def cost_rule(b, i):
            return 0.0
        block.cost = pyo.Expression(model.T, rule=cost_rule)

