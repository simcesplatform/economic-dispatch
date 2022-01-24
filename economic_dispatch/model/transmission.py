# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for Line
"""

import pyomo.environ as pyo


class Line:

    def __init__(self, name, b0, b1, reactance=None):
        """

        :param name: str
            name of the line

        :param reactance: float
            line reactance
        """

        self.name = name
        self.reactance = reactance
        self.attached = (b0, b1)

    def block_rule(self, block):
        """

        :param block:
        :return:
        """

        model = block.model()

        def x_init(b):
            return self.reactance
        block.x = pyo.Param(initialize=x_init)
        block.power_flow = pyo.Var(model.T)
        
    @classmethod
    def from_json(cls, line_json):
        return cls(**line_json)
