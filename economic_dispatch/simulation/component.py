# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

import os
import asyncio
import json

from tools.tools import FullLogger, load_environmental_variables

from economic_dispatch.simulation.economic_dispatch import EconomicDispatch
from economic_dispatch.simulation.economic_dispatch_flex import EconomicDispatchFlex


# Names of used environment variables:
# Economic dispatch:
# -----------------------------------------------------------------------------
SCENARIO_JSON = "SCENARIO_JSON"   # Optional initialisation for network model
DISPATCH_TOPIC = "DISPATCH_TOPIC"
RESOURCE_TYPE = "RESOURCE_TYPE"
DISPATCH_HORIZON = "DISPATCH_HORIZON"
DISPATCH_TIMESTEP = "DISPATCH_TIMESTEP"
DISPATCH_SOLVER = "DISPATCH_SOLVER"
COMMITMENT_TIME = "COMMITMENT_TIME"     # Optional if commitments to be used

# Storage resources:
# -----------------------------------------------------------------------------
RESOURCE_STATE_TOPIC = "RESOURCE_STATE_TOPIC"

# Price forecaster:
# ------------------------------------------------------------------------------
PRICE_FORECAST_STATE_TOPIC = "PRICE_FORECAST_STATE_TOPIC"

# Static time series resource:
# ------------------------------------------------------------------------------
RESOURCE_FORECAST_TOPIC = "RESOURCE_FORECAST_TOPIC"

# LFM
# ------------------------------------------------------------------------------
MARKET_ID = "MARKET_ID"
STATUS_TOPIC = "STATUS_TOPIC"
LFM_RESULT_TOPIC = "LFM_RESULT_TOPIC"
REQUEST_TOPIC = "REQUEST_TOPIC"
OFFER_TOPIC = "OFFER_TOPIC"
CUSTOMER_INFO_TOPIC = "CUSTOMER_INFO_TOPIC"
# Open offer option
SKIP_OPEN_OFFERS = "SKIP_OPEN_OFFERS"
    
LOGGER = FullLogger(__name__)


def create_component() -> EconomicDispatch:
    """
    Create a EconomicDispatch based on the initialization environment variables.
    """

    component = EconomicDispatch

    LOGGER.debug('Reading environmental variables')
    try:

        environment = load_environmental_variables(
            (SCENARIO_JSON, str, "{}"),
            (DISPATCH_HORIZON, str, "PT36H"),
            (DISPATCH_TIMESTEP, str, "PT1H"),
            (DISPATCH_SOLVER, str, "glpk"),
            (DISPATCH_TOPIC, str, "ResourceForecastState"),
            (RESOURCE_TYPE, str, "Dispatch"),
            (MARKET_ID, str, None),
            (RESOURCE_FORECAST_TOPIC, str, "ResourceForecastState.Load,ResourceForecastState.Generator"),
            (PRICE_FORECAST_STATE_TOPIC, str, "PriceForecastState"),
            (RESOURCE_STATE_TOPIC, str, "ResourceState.Storage"),
            (STATUS_TOPIC, str, "Status.Ready"),
            (LFM_RESULT_TOPIC, str, "LFMMarketResult"),
            (REQUEST_TOPIC, str, "Request"),
            (OFFER_TOPIC, str, "Offer"),
            (CUSTOMER_INFO_TOPIC, str, "Init.CIS.CustomerInfo"),
            (COMMITMENT_TIME, str, None),
            (SKIP_OPEN_OFFERS, str, None)
            )

        LOGGER.debug('Environmental variables read!')

        # if there is a file extension the string is assumed to be a file name
        if os.path.splitext(environment[SCENARIO_JSON])[1] == '.json':
            with open(environment[SCENARIO_JSON], 'r') as JSON:
                ed_json = json.load(JSON)
        else:
            ed_json = json.loads(environment[SCENARIO_JSON])

        component_kwargs = {
            "scenario": ed_json,
            "dispatch_horizon": environment[DISPATCH_HORIZON],
            "dispatch_timestep": environment[DISPATCH_TIMESTEP],
            "dispatch_solver": environment[DISPATCH_SOLVER],
            "dispatch_topic": environment[DISPATCH_TOPIC],
            "resource_type": environment[RESOURCE_TYPE],
            "commitment_time": environment[COMMITMENT_TIME]
            }

        suffix = ".#"
        param_topics = [
            *[topic + suffix for topic in environment[RESOURCE_FORECAST_TOPIC].split(",")],
            *[topic + suffix for topic in environment[PRICE_FORECAST_STATE_TOPIC].split(",")],
            *[topic + suffix for topic in environment[RESOURCE_STATE_TOPIC].split(",")],
        ]

        # maybe use some other way to indicate which component to use
        if environment[MARKET_ID] is not None:
            param_topics += [
                *[topic for topic in environment[STATUS_TOPIC].split(",")],
                *[topic + suffix for topic in environment[LFM_RESULT_TOPIC].split(",")],
                *[topic + suffix for topic in environment[REQUEST_TOPIC].split(",")],
                *[topic for topic in environment[CUSTOMER_INFO_TOPIC].split(",")],
            ]
            component_kwargs["offer_topic"] = environment[OFFER_TOPIC]
            component_kwargs["market_id"] = environment[MARKET_ID]

            component_kwargs["skip_open_offers"] = environment[SKIP_OPEN_OFFERS]
            if component_kwargs["skip_open_offers"] is None:
                component_kwargs["skip_open_offers"] = False
            elif isinstance(component_kwargs["skip_open_offers"], str):
                if component_kwargs["skip_open_offers"] == "False":
                    component_kwargs["skip_open_offers"] = False
                elif component_kwargs["skip_open_offers"] == "True":
                    component_kwargs["skip_open_offers"] = True
            # Else hopefully boolean and everything is ok

            component = EconomicDispatchFlex

        component_kwargs["param_topics"] = param_topics

    except Exception as Error:
        LOGGER.error('Error in component creation {}'.format(Error))

    return component(**component_kwargs)


async def start_component():
    """
    Create and start a EconomicDispatch component.
    """
    resource = create_component()
    await resource.start()
    while not resource.is_stopped:
        await asyncio.sleep(2)

if __name__ == '__main__':
    asyncio.run(start_component())
