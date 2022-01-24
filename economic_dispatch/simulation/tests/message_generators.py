# -*- coding: utf-8 -*-
"""
Message generators for EconomicDispatch component testing.
"""

from typing import List, Union
from collections import namedtuple

from tools.tests.components import MessageGenerator

from tools.message.status import StatusMessage
from tools.message.block import TimeSeriesBlock, ValueArrayBlock, QuantityBlock

from domain_messages.resource import ResourceStateMessage
from domain_messages.price_forecaster import PriceForecastStateMessage
from domain_messages.resource_forecast import ResourceForecastPowerMessage
from domain_messages.InitCISCustomerInfo import InitCISCustomerInfoMessage
from domain_messages.Offer import OfferMessage
from domain_messages.Request import RequestMessage
from domain_messages.LFMMarketResult import LFMMarketResultMessage
from domain_messages.dispatch import ResourceForecastStateDispatchMessage

from economic_dispatch.simulation.dataclasses import Request, LFMResult

# Maybe use ready-made procemplus blocks etc instead
ResourceForecast = namedtuple('ResourceForecast', ['resource_name', 'real_power', 'time_index', 'unit_of_measure'])
PriceForecast = namedtuple('PriceForecast', ['market_id', 'price', 'time_index', 'unit_of_measure'])
Quantity = namedtuple('Quantity', ['value', 'unit_of_measure'])
StorageResourceState = namedtuple('StorageResourceState',
                                  ['resource_name', 'bus', 'real_power', 'reactive_power', 'state_of_charge'])

Dispatch = namedtuple('Dispatch', ['real_power', 'time_index', 'unit_of_measure'])
ResourceForecastStateDispatch = namedtuple('ResourceForecastStateDispatch', ['dispatch'])


class ManagerMessageGenerator(MessageGenerator):
    '''
    Custom generator for simulation manager messages.
    '''

    def __init__(self, simulation_id: str, process_id: str):
        '''
        Create a message generator with 60 minute epoch length.
        '''
        super().__init__(simulation_id, process_id)
        self.epoch_interval = 3600


class ResourceForecastPowerMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected ResourceForecastState messages."""

    def __init__(self, simulation_id: str, process_id: str, forecasts: List):
        super().__init__(simulation_id, process_id)
        self.forecasts = forecasts

    def get_resource_forecast_state_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[
        ResourceForecastPowerMessage, None]:
        """Get the expected ResourceForecastPowerMessage for the given epoch."""
        if epoch_number == 0 or epoch_number >= len(self.forecasts):
            return None

        # get the resource forecast state for this epoch.
        forecast = self.forecasts[epoch_number]

        self.latest_message_id = next(self.id_generator)
        forecast_block = TimeSeriesBlock(TimeIndex=forecast.time_index,
                                         Series={
                                             "RealPower": ValueArrayBlock(
                                                 UnitOfMeasure=forecast.unit_of_measure,
                                                 Values=forecast.real_power)
                                         }
                                         )

        resource_forecast_state_message = ResourceForecastPowerMessage(**{
            "Type": "ResourceForecastState.Power",
            "SimulationId": self.simulation_id,
            "SourceProcessId": self.process_id,
            "MessageId": self.latest_message_id,
            "EpochNumber": epoch_number,
            "TriggeringMessageIds": triggering_message_ids,
            "ResourceId": forecast.resource_name,
            "Forecast": forecast_block
        })

        return resource_forecast_state_message


class PriceForecastMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected PriceForecastState messages."""

    def __init__(self, simulation_id: str, process_id: str, forecasts: List):
        super().__init__(simulation_id, process_id)
        self.forecasts = forecasts

    def get_price_forecast_state_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[
        PriceForecastStateMessage, None]:
        """Get the expected PriceForecastStateMessage for the given epoch."""
        if epoch_number == 0 or epoch_number >= len(self.forecasts):
            return None

        # get the forecast state for this epoch.
        forecast = self.forecasts[epoch_number]

        self.latest_message_id = next(self.id_generator)
        forecast_block = TimeSeriesBlock(TimeIndex=forecast.time_index,
                                         Series={
                                             "Price": ValueArrayBlock(
                                                 UnitOfMeasure=forecast.unit_of_measure,
                                                 Values=forecast.price)
                                         }
                                         )

        price_forecast_state_message = PriceForecastStateMessage(**{
            "Type": "PriceForecastState",
            "SimulationId": self.simulation_id,
            "SourceProcessId": self.process_id,
            "MessageId": self.latest_message_id,
            "EpochNumber": epoch_number,
            "TriggeringMessageIds": triggering_message_ids,
            "MarketId": forecast.market_id,
            "Prices": forecast_block,
            "ResourceId": "",
            "PricingType": ""
        })

        return price_forecast_state_message


class ResourceStateMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected ResourceState messages."""

    def __init__(self, simulation_id: str, process_id: str, states: List):
        super().__init__(simulation_id, process_id)
        self.states = states

    def get_resource_state_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[
        ResourceStateMessage, None]:
        """Get the expected message for the given epoch."""
        if epoch_number == 0 or epoch_number >= len(self.states):
            return None

        # get the resource state for this epoch.
        state = self.states[epoch_number]

        self.latest_message_id = next(self.id_generator)
        real_power_block = QuantityBlock(Value=state.real_power.value, UnitOfMeasure=state.real_power.unit_of_measure)
        reactive_power_block = QuantityBlock(Value=state.reactive_power.value,
                                             UnitOfMeasure=state.reactive_power.unit_of_measure)
        state_of_charge_block = QuantityBlock(Value=state.state_of_charge.value,
                                              UnitOfMeasure=state.state_of_charge.unit_of_measure)

        resource_state_message = ResourceStateMessage(**{
            "Type": "ResourceState",
            "SimulationId": self.simulation_id,
            "SourceProcessId": self.process_id,
            "MessageId": self.latest_message_id,
            "EpochNumber": epoch_number,
            "TriggeringMessageIds": triggering_message_ids,
            "Bus": state.bus,
            "RealPower": real_power_block,
            "ReactivePower": reactive_power_block,
            "StateOfCharge": state_of_charge_block,
            "CustomerId": "nada"
        })

        return resource_state_message


class DispatchMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected ResourceForecastStateDispatchMessage messages."""

    def __init__(self, simulation_id: str, process_id: str, dispatches: List):
        super().__init__(simulation_id, process_id)
        self.dispatches = dispatches

    def get_dispatch_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[
        ResourceForecastStateDispatchMessage, None]:
        """Get the expected message for the given epoch."""
        if epoch_number == 0 or epoch_number >= len(self.dispatches):
            return None

        dispatch = self.dispatches[epoch_number]

        dispatch_block = {}
        for name, b in dispatch.dispatch.items():
            series = {"RealPower": ValueArrayBlock(Values=b.real_power, UnitOfMeasure=b.unit_of_measure)}
            dispatch_block[name] = TimeSeriesBlock(TimeIndex=b.time_index, Series=series)

        dispatch_message = ResourceForecastStateDispatchMessage(
            SimulationId=self.simulation_id,
            Type=ResourceForecastStateDispatchMessage.CLASS_MESSAGE_TYPE,
            SourceProcessId=self.process_id,
            MessageId=self.latest_message_id,
            EpochNumber=epoch_number,
            TriggeringMessageIds=triggering_message_ids,
            Dispatch=dispatch_block
        )

        return dispatch_message


class RequestMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected ResourceForecastStateDispatchMessage
    messages. """

    def __init__(self, simulation_id: str, process_id: str, requests: List):
        super().__init__(simulation_id, process_id)
        self.requests = requests

    def get_request_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[RequestMessage, None]:
        """Get the expected message for the given epoch."""
        if epoch_number == 0 or epoch_number >= len(self.requests):
            return None

        req = self.requests[epoch_number]
        if req is None:
            req_message = None
        else:
            req_message = RequestMessage(**{
                "Type": "Request",
                "SimulationId": self.simulation_id,
                "SourceProcessId": self.process_id,
                "MessageId": self.latest_message_id,
                "EpochNumber": epoch_number,
                "TriggeringMessageIds": triggering_message_ids,
                "ActivationTime": req.activation_time,
                "Duration": req.duration,
                "Direction": req.direction,
                "RealPowerMin": req.real_power_min,
                "RealPowerRequest": req.real_power_request,
                "CustomerIds": req.customer_ids,
                "CongestionId": req.congestion_id,
                "BidResolution": req.bid_resolution,
            })

        return req_message


class OfferMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected ResourceForecastStateDispatchMessage messages."""

    def __init__(self, simulation_id: str, process_id: str, offers: List):
        super().__init__(simulation_id, process_id)
        self.offers = offers

    def get_offer_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[OfferMessage, None]:
        """Get the expected message for the given epoch."""
        if epoch_number == 0 or epoch_number >= len(self.offers):
            return None

        # req = self.offers[epoch_number]

        return None


class LFMMarketResultMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected ResourceForecastStateDispatchMessage messages."""

    def __init__(self, simulation_id: str, process_id: str, results: List):
        super().__init__(simulation_id, process_id)
        self.results = results

    def get_lfm_result_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[
        LFMMarketResultMessage, None]:
        """Get the expected message for the given epoch."""
        if epoch_number == 0 or epoch_number >= len(self.results):
            return None

        results = self.results[epoch_number]
        if results is None or not results:
            results = [LFMResult(congestion_id=None, customer_ids=None, activation_time=None, duration=None,
                                 direction=None, real_power=None, price=None, offer_id=None)]
            result_count = 0
        else:
            result_count = len(results)

        result_messages = []
        for result in results:
            if not isinstance(result.real_power, TimeSeriesBlock) and result.real_power is not None:
                real_power_ts_block = TimeSeriesBlock([result.activation_time],
                                                      {"Regulation": ValueArrayBlock([result.real_power], "kW")})
            else:
                real_power_ts_block = result.real_power

            result_message = LFMMarketResultMessage(**{
                "Type": "LFMMarketResult",
                "SimulationId": self.simulation_id,
                "SourceProcessId": self.process_id,
                "MessageId": self.latest_message_id,
                "EpochNumber": epoch_number,
                "TriggeringMessageIds": triggering_message_ids,
                "ActivationTime": result.activation_time,
                "Duration": result.duration,
                "Direction": result.direction,
                "RealPower": real_power_ts_block,
                "Price": result.price,
                "CongestionId": result.congestion_id,
                "CustomerIds": result.customer_ids,
                "OfferId": result.offer_id,
                "ResultCount": result_count,
            })
            result_messages.append(result_message)

        return result_messages


class ReadyMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected Ready messages."""

    def get_ready_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[StatusMessage, None]:
        """Get the expected message for the given epoch."""
        if epoch_number == 0:
            return None

        message = StatusMessage(**{
            "Type": StatusMessage.CLASS_MESSAGE_TYPE,
            "SimulationId": self.simulation_id,
            "SourceProcessId": self.process_id,
            "MessageId": self.latest_message_id,
            "EpochNumber": epoch_number,
            "TriggeringMessageIds": triggering_message_ids,
            "Value": "ready",
        })
        return message


class CustomerInfoMessageGenerator(MessageGenerator):
    """Message generator for the tests. extended to produce the expected Ready messages."""

    def __init__(self, simulation_id: str, process_id: str, resource_id: List, customer_id: List):
        super().__init__(simulation_id, process_id)
        self.resource_id = resource_id
        self.customer_id = customer_id

    def get_customer_info_message(self, epoch_number: int, triggering_message_ids: List[str]) -> Union[
        InitCISCustomerInfoMessage, None]:
        """Get the expected message for the given epoch."""
        if epoch_number != 1:
            return None

        message = InitCISCustomerInfoMessage(**{
            "Type": "Init.CIS.CustomerInfo",
            "SimulationId": self.simulation_id,
            "SourceProcessId": self.process_id,
            "MessageId": self.latest_message_id,
            "EpochNumber": epoch_number,
            "TriggeringMessageIds": triggering_message_ids,
            "ResourceId": self.resource_id,
            "CustomerId": self.customer_id,
            "BusName": [""],
        })

        return message


class InputMessageGenerator:
    generator_types = [
        (ResourceForecastPowerMessageGenerator, 'get_resource_forecast_state_message'),
        (PriceForecastMessageGenerator, 'get_price_forecast_state_message'),
        (ResourceStateMessageGenerator, 'get_resource_state_message'),
        (RequestMessageGenerator, 'get_request_message'),
        (LFMMarketResultMessageGenerator, 'get_lfm_result_message'),
        (ReadyMessageGenerator, 'get_ready_message'),
        (CustomerInfoMessageGenerator, 'get_customer_info_message'),
    ]

    def __init__(self, generators: List):
        self.generators = generators

    def get_generating_function(self, generator):
        for gen, fun_name in self.generator_types:
            if isinstance(generator, gen):
                return getattr(generator, fun_name)
        return None

    def get_input_messages(self, epoch_number: int, triggering_message_ids: List[str]) -> None:
        """Get the expected input messages that Dispatch component expects"""
        messages = []
        for generator in self.generators:
            message = self.get_generating_function(generator)(epoch_number, triggering_message_ids)
            if isinstance(message, list):
                messages.extend(message)
            else:
                messages.append(message)
        return messages


HORIZON = 3

TIME_INDEX = [
    "2020-01-01T00:00:00.000Z",
    "2020-01-01T01:00:00.000Z",
    "2020-01-01T02:00:00.000Z",
    "2020-01-01T03:00:00.000Z",
    "2020-01-01T04:00:00.000Z",
    "2020-01-01T05:00:00.000Z",
    "2020-01-01T06:00:00.000Z",
]

BATT_DISPATCH_VALUES = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
MARKET_DISPATCH_VALUES = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

GEN_FORECAST_VALUES = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
LOAD_FORECAST_VALUES = [-20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0]
PRICE_FORECAST_VALUES = [10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0]
STORAGE_STATE_VALUES = [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0]

# Customer id: id1 -> "Battery"

# Epoch 1: Iteration over values from min:bid:req too large -> Lowers max req
_REQUEST = Request(congestion_id="cid1", customer_ids=["id1"], activation_time="2020-01-01T02:00:00.000Z", duration=60,
                   direction="upregulation", real_power_min=0.5, real_power_request=0.5, bid_resolution=1.0)
# Working offer
# One open offer taken into account from previous epoch
_REQUEST2 = Request(congestion_id="cid2", customer_ids=["id1", "SomethingNotHere"],
                    activation_time="2020-01-01T03:00:00.000Z", duration=60,
                    direction="downregulation", real_power_min=1.0, real_power_request=6.0, bid_resolution=1.0)
# Working offer
_REQUEST3 = Request(congestion_id="cid3", customer_ids=["id1"], activation_time="2020-01-01T03:00:00.000Z", duration=60,
                    direction="downregulation", real_power_min=1.0, real_power_request=5.4, bid_resolution=1.0)
# Min power too large -> zero offer
_REQUEST4 = Request(congestion_id="cid4", customer_ids=["id1"], activation_time="2020-01-01T03:00:00.000Z", duration=60,
                    direction="downregulation", real_power_min=100.0, real_power_request=100.0, bid_resolution=1.0)
# Customer id not in network -> zero offer
_REQUEST5 = Request(congestion_id="cid5", customer_ids=["SomethingNotHere"], activation_time="2020-01-01T04:00:00.000Z",
                    duration=60,
                    direction="downregulation", real_power_min=1.0, real_power_request=1.0, bid_resolution=1.0)
REQUESTS = [None, _REQUEST, _REQUEST2, _REQUEST3, _REQUEST4, _REQUEST5, None, None]

# Not tested here
# Congestion id already found -> zero offer

# Epoch 1: zero results
# Epoch 2: zero results
# Epoch 3: accept 2 offers from epoch 2
_LFM_RESULT = LFMResult(congestion_id="cid2", customer_ids=["id1"], activation_time="2020-01-01T02:00:00.000Z",
                        duration=60,
                        direction="downregulation", real_power=1.0, price=1.0, offer_id="ED-test-flex-cid2-1")
_LFM_RESULT3 = LFMResult(congestion_id="cid2", customer_ids=["id1"], activation_time="2020-01-01T02:00:00.000Z",
                         duration=60,
                         direction="downregulation", real_power=1.0, price=1.0, offer_id="ED-test-flex-cid2-3")
LFM_RESULTS = [None, None, None, [_LFM_RESULT, _LFM_RESULT3], [_LFM_RESULT3], None, None, None]


def generate_data():
    # Epoch 0
    dispatches = [None]
    not_our_gen_forecasts = [None]
    gen_forecasts = [None]
    load_forecasts = [None]
    price_forecasts = [None]
    storage_states = [None]

    for i, _ in enumerate(TIME_INDEX[:-(HORIZON - 1)]):
        time_index = TIME_INDEX[i:i + HORIZON]

        battery = Dispatch(real_power=BATT_DISPATCH_VALUES[i:i + HORIZON], time_index=time_index, unit_of_measure="kW")
        market = Dispatch(real_power=MARKET_DISPATCH_VALUES[i:i + HORIZON], time_index=time_index, unit_of_measure="kW")
        d = {"Battery": battery, "ElectricityMarket": market}
        dispatches.append(ResourceForecastStateDispatch(dispatch=d))

        not_our_gen_forecasts.append(
            ResourceForecast(resource_name="CHP2", real_power=GEN_FORECAST_VALUES[i:i + HORIZON], time_index=time_index,
                             unit_of_measure="kW"))
        gen_forecasts.append(
            ResourceForecast(resource_name="CHP", real_power=GEN_FORECAST_VALUES[i:i + HORIZON], time_index=time_index,
                             unit_of_measure="kW"))
        load_forecasts.append(ResourceForecast(resource_name="Demand", real_power=LOAD_FORECAST_VALUES[i:i + HORIZON],
                                               time_index=time_index, unit_of_measure="kW"))
        price_forecasts.append(PriceForecast(market_id="ElectricityMarket", price=PRICE_FORECAST_VALUES[i:i + HORIZON],
                                             time_index=time_index, unit_of_measure="{EUR}/(kW.h)"))

        real_power = Quantity(0.0, "kW")
        reactive_power = Quantity(0.0, "kV.A{r}")
        soc = Quantity(STORAGE_STATE_VALUES[i], "%")
        storage_states.append(StorageResourceState(resource_name="Battery", bus="b0", real_power=real_power,
                                                   reactive_power=reactive_power, state_of_charge=soc))

    return dispatches, gen_forecasts, load_forecasts, price_forecasts, storage_states, not_our_gen_forecasts


DISPATCHES, GEN_FORECASTS, LOAD_FORECASTS, PRICE_FORECASTS, STORAGE_STATES, NOT_OUR_GEN_FORECASTS = generate_data()
