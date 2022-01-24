# -*- coding: utf-8 -*-
'''
Tests for the EconomicDispatch component.
'''

import asyncio
from typing import List, Tuple, Union, cast
import os
# from collections import namedtuple
# import isodate
import json

import unittest
#import aiounittest

from tools.clients import RabbitmqClient
from tools.components import AbstractSimulationComponent, SIMULATION_START_MESSAGE_FILENAME
from tools.tests.components import MessageGenerator, TestAbstractSimulationComponent, MessageStorage, send_message
from tools.tests.components import TestAbstractSimulationComponent

from tools.messages import AbstractMessage
# from tools.message.block import TimeSeriesBlock, ValueArrayBlock, QuantityBlock

from domain_messages.resource import ResourceStateMessage
from domain_messages.price_forecaster import PriceForecastStateMessage
from domain_messages.resource_forecast import ResourceForecastPowerMessage

from economic_dispatch.simulation.component import create_component, SCENARIO_JSON, RESOURCE_TYPE, \
    RESOURCE_STATE_TOPIC, PRICE_FORECAST_STATE_TOPIC, RESOURCE_FORECAST_TOPIC, DISPATCH_HORIZON, DISPATCH_SOLVER
from domain_messages.dispatch import ResourceForecastStateDispatchMessage
# from economic_dispatch.simulation.message import DispatchBlock, ResourceForecastStateDispatchMessage

from economic_dispatch.simulation.tests.message_generators import ManagerMessageGenerator, ResourceForecastPowerMessageGenerator, \
    PriceForecastMessageGenerator, ResourceStateMessageGenerator, DispatchMessageGenerator, InputMessageGenerator

# test data
from economic_dispatch.simulation.tests.message_generators import DISPATCHES, GEN_FORECASTS, LOAD_FORECASTS, \
    STORAGE_STATES, PRICE_FORECASTS, NOT_OUR_GEN_FORECASTS


SIMULATION_EPOCHS = len(DISPATCHES)-1
SIMULATION_ID= "2020-01-01T00:00:00.000Z"
DEMO_INPUT_GENERATORS = [
    ResourceForecastPowerMessageGenerator(simulation_id=SIMULATION_ID, process_id="CHP", forecasts=GEN_FORECASTS),
    ResourceForecastPowerMessageGenerator(simulation_id=SIMULATION_ID, process_id="CHP2", forecasts=NOT_OUR_GEN_FORECASTS),
    ResourceForecastPowerMessageGenerator(simulation_id=SIMULATION_ID, process_id="Demand", forecasts=LOAD_FORECASTS),
    PriceForecastMessageGenerator(simulation_id=SIMULATION_ID, process_id="Market", forecasts=PRICE_FORECASTS),
    ResourceStateMessageGenerator(simulation_id=SIMULATION_ID, process_id="Battery", states=STORAGE_STATES),
]

ED_COMPONENT_NAME = "ED-test"

START = {
    "ProcessParameters": {
        "EconomicDispatch": {
            ED_COMPONENT_NAME: {
                "Horizon": "PT3H",
                "Timestep": "PT1H",
                "Resources": [
                    ["StaticTimeSeriesResource", "Demand"],
                    ["StaticTimeSeriesResource", "CHP"],
                    ["PriceForecaster", "ElectricityMarket"],
                    ["StorageResource", "Battery"]
                ],
                "Weights": {
                    "default": {
                        "TerminalSOCBound": 0.4,
                        "TerminalSOCTarget": 0.5,
                        "TerminalWeight": None,
                    },
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
                "KwRated": 10.0,
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
            "max_power": 10.0,
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

# specify component initialization environment variables.
os.environ[SCENARIO_JSON] = json.dumps(TEST_SCENARIO)
os.environ[RESOURCE_TYPE] = str('Dispatch')
os.environ[DISPATCH_HORIZON] = str('PT3H')
os.environ[DISPATCH_SOLVER] = str('cbc')
os.environ[RESOURCE_FORECAST_TOPIC] = str("ResourceForecastState.Load,ResourceForecastState.Generator")
os.environ[PRICE_FORECAST_STATE_TOPIC] = str("PriceForecastState")
os.environ[RESOURCE_STATE_TOPIC] = str("ResourceState.Storage")

start_fname = "test_start.json"
with open(start_fname, 'w') as JSON:
    json.dump(START, JSON)

os.environ[SIMULATION_START_MESSAGE_FILENAME] = str(start_fname)


class TestEconomicDispatchComponent(TestAbstractSimulationComponent):
    """Unit tests for EconomicDispatch component.""" 

    simulation_id = SIMULATION_ID
    component_name = ED_COMPONENT_NAME

    short_wait = 1.0
    long_wait = 4.0

    # the method which initializes the component
    component_creator = create_component

    message_generator_type = DispatchMessageGenerator
    additional_generator_params = {"dispatches": DISPATCHES}

    input_message_generator_type = InputMessageGenerator
    input_generators = DEMO_INPUT_GENERATORS

    normal_simulation_epochs = SIMULATION_EPOCHS
    # use custom manager whose epoch length matches the test data.
    test_manager_name = "TestManager"
    manager_message_generator = ManagerMessageGenerator(simulation_id, test_manager_name)

    def get_input_messages(self, input_message_generator: InputMessageGenerator, epoch_number: int,
                           triggering_message_ids: List[str]) -> List[Tuple[AbstractMessage, str]]:
        """Get the messages and topics the component is expected to publish in given epoch."""
        if epoch_number == 0:
            return []
            
        messages = input_message_generator.get_input_messages(epoch_number, triggering_message_ids)
        topics = []
        for msg in messages:
            if isinstance(msg, ResourceForecastPowerMessage):
                topic = "ResourceForecastState.Load." + msg.source_process_id
            elif isinstance(msg, PriceForecastStateMessage):
                topic = os.environ[PRICE_FORECAST_STATE_TOPIC] + ".TestMarket"
            elif isinstance(msg, ResourceStateMessage):
                topic = os.environ[RESOURCE_STATE_TOPIC] + "." + msg.source_process_id
            elif msg is None:
                topic = None
            else:
                raise TypeError

            topics.append(topic)
        
        return [(m, t) for m, t in zip(messages, topics) if m is not None]
    
    def get_expected_messages(self, component_message_generator: MessageGenerator, epoch_number: int, triggering_message_ids: List[str]) -> List[Tuple[AbstractMessage, str]]:
        """Get the messages and topics the component is expected to publish in given epoch."""
        if epoch_number == 0:
            return [
                (component_message_generator.get_status_message(epoch_number, triggering_message_ids), "Status.Ready")
                ]
            
        return [
            (component_message_generator.get_dispatch_message(epoch_number, triggering_message_ids), "ResourceForecastState." + os.environ[RESOURCE_TYPE]),
            (component_message_generator.get_status_message(epoch_number, triggering_message_ids), "Status.Ready")
        ]
    
    async def start_tester(self) -> Tuple[RabbitmqClient, MessageStorage,
                                          MessageGenerator, InputMessageGenerator, AbstractSimulationComponent]:
        """Tests the creation of the test component at the start of the simulation and returns a 5-tuple containing
           the message bus client, message storage object, test component message generator object,
           test input message generator object and the test component object for the use of further tests."""

        await asyncio.sleep(self.__class__.long_wait)

        message_storage = MessageStorage(self.__class__.test_manager_name)
        message_client = RabbitmqClient()
        message_client.add_listener("#", message_storage.callback)

        input_message_generator = self.__class__.input_message_generator_type(self.input_generators)

        component_message_generator = self.__class__.message_generator_type(
            self.__class__.simulation_id, self.__class__.component_name, **self.additional_generator_params)
        test_component = self.__class__.component_creator(**self.__class__.component_creator_params)
        await test_component.start()

        # Wait for a few seconds to allow the component to setup.
        await asyncio.sleep(self.__class__.short_wait)
        self.assertFalse(message_client.is_closed)
        self.assertFalse(test_component.is_stopped)
        self.assertFalse(test_component.is_client_closed)

        self.assertEqual(test_component.simulation_id, self.__class__.simulation_id)
        self.assertEqual(test_component.component_name, self.__class__.component_name)

        return (message_client, message_storage, component_message_generator, input_message_generator, test_component)

    async def epoch_tester(self, epoch_number: int, test_component: AbstractSimulationComponent, message_client: RabbitmqClient,
                           message_storage: MessageStorage, component_message_generator: MessageGenerator, 
                           input_message_generator: MessageGenerator):
        """Test the behaviour of the test_component in one epoch."""

        number_of_previous_messages = len(message_storage.messages_and_topics)
        if epoch_number == 0:
            # Epoch number 0 corresponds to the start of the simulation.
            manager_message = self.__class__.manager_message_generator.\
                get_simulation_state_message(True)
        else:
            manager_message = self.__class__.manager_message_generator.get_epoch_message(
                epoch_number, [component_message_generator.latest_message_id])
        component_inputs = self.get_input_messages(
            input_message_generator, epoch_number, [manager_message.message_id])
        expected_responds = self.get_expected_messages(
            component_message_generator, epoch_number, [manager_message.message_id])

        await send_message(message_client, manager_message, manager_message.message_type)

        # Clear epoch variables

        for input_message, topic_name in component_inputs:
            await send_message(message_client, input_message, topic_name)

        # Wait to allow the message storage to store the respond. Wait epoch processing? (not ready)
        #if epoch_number != 0:
        #    await test_component._event.wait()
        await asyncio.sleep(self.__class__.long_wait)

        received_messages = message_storage.messages_and_topics
        self.assertEqual(len(received_messages), number_of_previous_messages + len(component_inputs) + len(expected_responds))

        # Compare the received messages to the expected messages.
        # TODO: implement message checking that does not care about the order of the received messages
        for index, (received_responce, expected_responce) in enumerate(
                zip(received_messages[-len(expected_responds):], expected_responds),
                start=1):
            with self.subTest(message_index=index):
                received_message, received_topic = received_responce
                expected_message, expected_topic = expected_responce
                self.assertEqual(received_topic, expected_topic)
                self.assertTrue(self.compare_message(received_message, expected_message))

    async def test_normal_simulation(self):
        """A test with a normal input in a simulation containing only manager and test component."""
        # Test the creation of the test component.
        message_client, message_storage, component_message_generator, input_message_generator, test_component = \
            await self.start_tester()

        # Test the component with the starting simulation state message (epoch 0) and n normal epochs.
        for epoch_number in range(0, self.__class__.normal_simulation_epochs + 1):
            with self.subTest(epoch_number=epoch_number):
                await self.epoch_tester(epoch_number, test_component, message_client, message_storage, component_message_generator, input_message_generator)

        # Test the closing down of the test component after simulation state message "stopped".
        await self.end_tester(message_client, test_component)

    @unittest.skip("not implemented")
    async def test_error_message(self):
        """Unit test for simulation component sending an error message."""
        # Setup the component and start the simulation.
        """
        message_client, message_storage, component_message_generator, input_message_generator, test_component = \
            await self.start_tester()
        await self.epoch_tester(0, message_client, message_storage, component_message_generator, input_message_generator)

        # Generate the expected error message and check if it matches to the one the test component sends.
        error_description = "Testing error message"
        expected_message = component_message_generator.get_error_message(
            epoch_number=0,
            triggering_message_ids=[self.__class__.manager_message_generator.latest_message_id],
            description=error_description)
        number_of_previous_messages = len(message_storage.messages_and_topics)
        await test_component.send_error_message(error_description)

        # Wait a short time to ensure that the message receiver has received the error message.
        await asyncio.sleep(self.__class__.short_wait)

        # Check that the correct error message was received.
        self.assertEqual(len(message_storage.messages_and_topics), number_of_previous_messages + 1)
        received_message, received_topic = message_storage.messages_and_topics[-1]
        self.assertEqual(received_topic, "Status.Error")
        self.assertTrue(self.compare_message(received_message, expected_message))

        await self.end_tester(message_client, test_component) """
        return

    def compare_dispatch_message(self, first_message: ResourceForecastStateDispatchMessage, second_message: ResourceForecastStateDispatchMessage):
        """Check that the two ResourceForecastPower messages have the same content."""
        self.compare_abstract_result_message(first_message, second_message)
        
        # block checks
        first_dispatch_block = first_message.dispatch
        second_dispatch_block = second_message.dispatch
        
        for name in first_dispatch_block.keys():
            self.assertTrue(name in second_dispatch_block.keys())
            
            # component dispatch time series blocks
            first_ts_block = first_dispatch_block[name]
            second_ts_block = second_dispatch_block[name]

            self.assertListEqual(first_ts_block.series['RealPower'].values, second_ts_block.series['RealPower'].values )
            self.assertListEqual(first_ts_block.time_index, second_ts_block.time_index)
            self.assertEqual(first_ts_block.series['RealPower'].unit_of_measure, second_ts_block.series['RealPower'].unit_of_measure )
        
    def compare_message(self, first_message: AbstractMessage, second_message: AbstractMessage) -> bool:
        """Override the super class implementation to add the comparison of ResourceForecastStateDispatchMessage messages."""
        if super().compare_message(first_message, second_message):
            return True

        if isinstance(second_message, ResourceForecastStateDispatchMessage):
            self.compare_dispatch_message(cast(ResourceForecastStateDispatchMessage, first_message), second_message)
            return True

        return False

# To skip baseclass tests
del TestAbstractSimulationComponent