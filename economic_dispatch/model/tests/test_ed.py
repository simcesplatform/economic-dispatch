# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Tests for the ResourceForecastStateDispatchMessage class.
"""
import os
import unittest
import json

import pyomo.environ as pyo

from economic_dispatch.model.network import Network
from economic_dispatch.model.bus import Bus
from economic_dispatch.model.transmission import Line
from economic_dispatch.model.units import StaticUnit, StorageUnit, Retailer

from economic_dispatch.planner.ed import EconomicDispatchPlanner
from economic_dispatch.planner.exceptions import ModelBuildError
from economic_dispatch.model.factory import UnitFactory


class TestModelCreation(unittest.TestCase):

    def test_unit_creation(self):
        staticload = StaticUnit('Demand')
        self.assertIsInstance(staticload, StaticUnit)

        staticgenerator = StaticUnit('CHP')
        self.assertIsInstance(staticgenerator, StaticUnit)

        battery = StorageUnit('Battery', 100.0, 10.0)
        self.assertIsInstance(battery, StorageUnit)

        market = Retailer('ElectricityMarket')
        self.assertIsInstance(market, Retailer)

    def test_unit_factory(self):
        staticload = UnitFactory.make_component('staticunit', name='Demand')
        self.assertIsInstance(staticload, StaticUnit)

        staticgenerator = UnitFactory.make_component('staticunit', name='CHP')
        self.assertIsInstance(staticgenerator, StaticUnit)

        battery = UnitFactory.make_component('storage', name='Battery', capacity=100.0, max_power=10.0)
        self.assertIsInstance(battery, StorageUnit)

        market = UnitFactory.make_component('market', name='ElectricityMarket')
        self.assertIsInstance(market, Retailer)

    def test_bus_creation(self):
        staticload = StaticUnit('Demand')
        staticgenerator = StaticUnit('CHP')
        battery = StorageUnit('Battery', 100.0, 10.0)
        market = Retailer('ElectricityMarket')
        
        units = [staticload, staticgenerator, battery, market]
        bus = Bus('bus0', units=units)
        self.assertIsInstance(bus, Bus)

        units2 = [staticload, staticgenerator, battery]
        units2_names = [u.name for u in units2]
        bus2 = Bus('bus1', units=units2)

        bus2.add(market)
        self.assertEqual(set(bus.unit_names), set(bus2.unit_names))

        bus2.remove(market.name)
        self.assertEqual(set(bus2.unit_names), set(units2_names))

        self.assertRaises(ValueError, bus2.add, 'not_a_unit')

    def test_network_creation(self):
        staticload = StaticUnit('Demand')
        staticgenerator = StaticUnit('CHP')
        battery = StorageUnit('Battery', 100.0, 10.0)
        market = Retailer('ElectricityMarket')
        
        units = [staticload, staticgenerator, battery, market]
        bus = Bus('bus0', units=units)

        network = Network(name='Network', buses=[bus], lines=[])
        self.assertIsInstance(network, Network)

        network2 = Network(name='Network', buses=[], lines=[])
        self.assertIsInstance(network, Network)

        network2.add_bus(bus)
        self.assertEqual(set(network.bus_names), set(network2.bus_names))

        network2.remove_bus(bus.name)
        self.assertEqual(set(network2.bus_names), set())

        self.assertRaises(ValueError, network2.add_bus, 'not_a_bus')

    def test_from_json(self):
        unames = ['CHP', 'Battery', 'ElectricityMarket', 'Demand']
        units_fname = os.path.join(os.path.dirname(__file__), 'units.json')
        with open(units_fname, 'r') as JSON:
            nw_json = json.load(JSON)

        network = Network.from_json(nw_json)
        self.assertIsInstance(network, Network)
        self.assertEqual(set(network.unit_names), set(unames))


class TestEconomicDispatchPlanner(unittest.TestCase):

    def test_planner(self):
        staticload = StaticUnit('Demand')
        staticgenerator = StaticUnit('CHP')
        battery = StorageUnit('Battery', 100.0, 10.0)
        market = Retailer('ElectricityMarket')
        
        units = [staticload, staticgenerator, battery, market]
        bus = Bus('bus0', units=units)

        network = Network(name='Network', buses=[bus], lines=[])

        ed = EconomicDispatchPlanner(network, horizon=3, solver='glpk')
        self.assertIsInstance(ed, EconomicDispatchPlanner)

        initial_state = 0.7 # %/100
        forecast_gen = [5.0, 5.0, 5.0]
        forecast_price = [1.0, 1.0, 1.0]
        forecast_load = [-20.0, -20.0, -20.0]

        self.assertFalse(ed.ready_to_solve())

        self.assertFalse(ed.units['CHP'].ready())
        ed.set_param('CHP', 'forecast', forecast_gen)
        self.assertTrue(ed.units['CHP'].ready())

        self.assertFalse(ed.ready_to_solve())

        self.assertFalse(ed.units['Battery'].ready())
        ed.set_param('Battery', 'state', initial_state)
        self.assertTrue(ed.units['Battery'].ready())

        self.assertFalse(ed.ready_to_solve())

        self.assertFalse(ed.units['ElectricityMarket'].ready())
        ed.set_param('ElectricityMarket', 'prices', forecast_price)
        self.assertTrue(ed.units['ElectricityMarket'].ready())

        self.assertFalse(ed.ready_to_solve())

        #self.assertRaises(ModelBuildError, ed.solve) # ok but error messages printed

        ed.set_param('Demand', 'forecast', forecast_load)

        # Last param set, ready now?
        self.assertTrue(ed.ready_to_solve())
        ed.build_model()

        ed.clear()
        self.assertFalse(ed.ready_to_solve())
        # Shallow clear does not clear parameter values so assertTrue
        self.assertTrue(any(ed.units[u].ready() for u in ed.unit_names))

        ed._shallow_clear = False
        ed.clear()
        # Now parameter values are cleared
        self.assertFalse(any(ed.units[u].ready() for u in ed.unit_names))
        self.assertFalse(ed.network.ready())
        self.assertFalse(ed.ready_to_solve())

    def test_from_json(self):
        unames = ['CHP', 'Battery', 'ElectricityMarket', 'Demand', 'Solar']
        ed_fname = os.path.join(os.path.dirname(__file__), 'scenario.json')
        with open(ed_fname, 'r') as JSON:
            ed_json = json.load(JSON)

        ed = EconomicDispatchPlanner.from_json(ed_json)
        self.assertIsInstance(ed, EconomicDispatchPlanner)
        self.assertEqual(set(ed.network.unit_names), set(unames))
        


if __name__ == "__main__":
    unittest.main()
