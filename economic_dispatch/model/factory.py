# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class _UnitFactory and creates one to UnitFactory.
"""

from economic_dispatch.model.units.static_units import StaticUnit
from economic_dispatch.model.units.storages import StorageUnit
from economic_dispatch.model.units.markets import Retailer


class _UnitFactory:

    DEFAULT_REGISTRY = {
        "staticunit": StaticUnit,
        "storage": StorageUnit,
        "market": Retailer,
    }

    def __init__(self):
        self.registry = self.DEFAULT_REGISTRY

    def register_component(self, key, component_class):
        self.registry[key] = component_class

    def make_component(self, component_key, **kwargs):
        component_class = self.registry[component_key]
        component = component_class(**kwargs)
        return component


UnitFactory = _UnitFactory()
