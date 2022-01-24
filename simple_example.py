# -*- coding: utf-8 -*-

"""
Contains a simple test.
"""
import pyomo.environ as pyo 

from economic_dispatch.model.units import (
    StorageUnit,
    StaticUnit,
    Retailer,
    )
from economic_dispatch.model.bus import Bus
from economic_dispatch.model.network import Network
from economic_dispatch.planner.ed import EconomicDispatchPlanner

import json


def test_storage():
    storage_unit = StorageUnit('Battery', 100.0, 10.0)
    initial_state = 0.7 # %/100

    static_generator = StaticUnit('CHP')
    forecast_gen = [5.0, 5.0, 5.0]

    retailer = Retailer('ElectricityMarket')
    forecast_price = [1.0, 1.0, 1.0]

    static_load = StaticUnit('Demand')
    forecast_load = [-20.0, -20.0, -20.0]

    units = [storage_unit, static_generator, retailer, static_load]
    bus = Bus('Bus0', units=units)

    network = Network(name='Network', buses=[bus], lines=[])

    ed = EconomicDispatchPlanner(network, horizon=3, solver='glpk')
    print('Problem created')

    ed.set_param('CHP', 'forecast', forecast_gen)
    ed.set_param('Battery', 'state', initial_state)
    ed.set_param('ElectricityMarket', 'prices', forecast_price)
    ed.set_param('Demand', 'forecast', forecast_load)

    print('Ready to solve: '+str(ed.ready_to_solve()))

    ed.solve()
    results = ed.results()
    print(json.dumps(results, indent=4))
    # "Battery" positive, if positive output towards the grid
    print("Cost: ", pyo.value(ed.model.price), '\n')

    comm = {1: 4.0}
    print("Solve with commitments {!s}".format(comm))
    print("Can't fulfill commitments, battery at max already, diverge from commitment with extra cost")
    ed.set_param('ElectricityMarket', 'commitments', comm)
    ed.solve()
    results = ed.results()
    print(json.dumps(results, indent=4))
    print("Cost: ", pyo.value(ed.model.price), '\n')

    comm = {1: 6.0}
    print("Solve with commitments {!s}".format(comm))
    print("Can fulfill commitments.")
    ed.set_param('ElectricityMarket', 'commitments', comm)
    ed.solve()
    results = ed.results()
    print(json.dumps(results, indent=4))
    print("Cost: ", pyo.value(ed.model.price), '\n')

    ed.clear()
    print('Cleared barrier.')

    # print('Ready to solve: '+str(ed.ready_to_solve()))


if __name__ == '__main__':
    test_storage()