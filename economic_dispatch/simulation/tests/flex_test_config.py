# -*- coding: utf-8 -*-
'''
Some data for tests.
'''

from economic_dispatch.simulation.tests.message_generators import ManagerMessageGenerator, ResourceForecastPowerMessageGenerator, \
    PriceForecastMessageGenerator, ResourceStateMessageGenerator, DispatchMessageGenerator, InputMessageGenerator, \
    RequestMessageGenerator, LFMMarketResultMessageGenerator, ReadyMessageGenerator, CustomerInfoMessageGenerator

# test data
from economic_dispatch.simulation.tests.message_generators import GEN_FORECASTS, LOAD_FORECASTS, \
    STORAGE_STATES, PRICE_FORECASTS, REQUESTS, LFM_RESULTS

SIMULATION_ID = "2020-01-01T00:00:00.000Z"
SIMULATION_EPOCHS = len(GEN_FORECASTS)-1


S1_INPUT_GENERATORS = [
    CustomerInfoMessageGenerator(simulation_id=SIMULATION_ID, process_id="LFM1", resource_id=["Battery","Haloo"],
                                 customer_id=["id1","SomethingNotHere"]),
    ResourceForecastPowerMessageGenerator(simulation_id=SIMULATION_ID, process_id="CHP", forecasts=GEN_FORECASTS),
    ResourceForecastPowerMessageGenerator(simulation_id=SIMULATION_ID, process_id="Demand", forecasts=LOAD_FORECASTS),
    PriceForecastMessageGenerator(simulation_id=SIMULATION_ID, process_id="Market", forecasts=PRICE_FORECASTS),
    LFMMarketResultMessageGenerator(simulation_id=SIMULATION_ID, process_id="LFM1", results=LFM_RESULTS),
    ResourceStateMessageGenerator(simulation_id=SIMULATION_ID, process_id="Battery", states=STORAGE_STATES),
]

S2_INPUT_GENERATORS = [
    RequestMessageGenerator(simulation_id=SIMULATION_ID, process_id="LFM1", requests=REQUESTS),
    ReadyMessageGenerator(simulation_id=SIMULATION_ID, process_id="LFM1"),
]

DEMO_INPUT_GENERATORS = [
    S1_INPUT_GENERATORS,
    S2_INPUT_GENERATORS,
]

ED_COMPONENT_NAME = "ED-test-flex"

START = {
    "ProcessParameters": {
        "EconomicDispatch": {
            ED_COMPONENT_NAME: {
                "Horizon": "PT3H",
                "Timestep": "PT1H",
                "SkipOpenOffers": "True",
                "Resources": [
                    ["StaticTimeSeriesResource", "Demand"],
                    ["StaticTimeSeriesResource", "CHP"],
                    ["PriceForecaster", "ElectricityMarket"],
                    ["StorageResource", "Battery"]
                ],
                "Weights": {
                    # "default": {
                    #     "TerminalSOCBound": 0.4,
                    #     "TerminalSOCTarget": 0.5,
                    #     "TerminalWeight": None,
                    # },
                    "Battery": {
                        "TerminalSOCBound": 0.3,
                    }
                }
            }
        },
        "StaticTimeSeriesResource": {
            "Demand": {
                "ResourceType": "Load",
                "ResourceStateFile": "load.csv",
                "ResourceFileDelimiter": ","
                },
            "CHP": {
                "ResourceType": "Generator",
                "ResourceStateFile": "gen.csv",
                "ResourceFileDelimiter": ","
                },
        },
        "PriceForecaster": {
            "ElectricityMarket": {
                "PriceForecasterStateCsvFile": "priceB.csv"
                },
        },
        "StorageResource": {
            "Battery": {
                "ResourceStateCsvFile": "control.csv",
                "ResourceStateCsvDelimiter": ",",
                "Bus": "bus1", 
                "KwhRated": 100.0,
                "KwRated": 50.0,
                "InitialStateOfCharge": 70.0
            }
        }
    }
}


TEST_SCENARIO = {
    "units": [
        {
            "component_key": "storage",
            "name": "Battery",
            "capacity": 100.0,
            "max_power": 50.0,
        },
        {
            "component_key": "staticload",
            "name": "Demand",
        },
        {
            "component_key": "staticgenerator",
            "name": "CHP",
        },
        {
            "component_key": "market",
            "name": "ElectricityMarket",
        },
    ]
}
