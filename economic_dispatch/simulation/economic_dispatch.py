# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for a simulation platform component used to optimize economic dispatch
"""

from typing import Union, Any, Dict, List
from datetime import timedelta, datetime
import json

from isodate import parse_duration, parse_datetime

from tools.tools import FullLogger
from tools.datetime_tools import to_iso_format_datetime_string, to_utc_datetime_object
from tools.components import AbstractSimulationComponent
from tools.message.abstract import BaseMessage
from tools.message.block import TimeSeriesBlock, ValueArrayBlock

from domain_messages.resource import ResourceStateMessage
from domain_messages.price_forecaster import PriceForecastStateMessage
from domain_messages.resource_forecast import ResourceForecastPowerMessage
from domain_messages.dispatch import ResourceForecastStateDispatchMessage

from economic_dispatch.planner.ed import EconomicDispatchPlanner
from economic_dispatch.model.network import Network
from economic_dispatch.simulation.start import load_start_parameters


LOGGER = FullLogger(__name__)


class EconomicDispatch(AbstractSimulationComponent):
    """
    A simulation platform component that optimizes power production/consumption over a given horizon given the prices
    and states.
    """
    PLANNER = EconomicDispatchPlanner

    def __init__(self, scenario: Union[str, Dict],
                 param_topics: List[str],
                 dispatch_horizon: str,
                 dispatch_timestep: str,
                 dispatch_solver: str,
                 dispatch_topic: str,
                 resource_type: str,
                 commitment_time: str,
                 initialization_error: str = None
                 ):

        super().__init__(other_topics=param_topics)

        self.initialization_error = initialization_error

        LOGGER.debug('Reading start message')

        # override provided values if start message file read
        if self.start_message is not None:
            try:
                dispatch_horizon, dispatch_timestep, scenario, init_states, start_error = \
                    load_start_parameters(self.start_message, self.component_name)
            except Exception as Error:
                LOGGER.error('Error in start message {}'.format(Error))

            # Start message errors
            if start_error:
                if self.initialization_error is not None:
                    self.initialization_error = self.initialization_error + \
                                                ' --- '.join(value for value in start_error.values())
                else:
                    self.initialization_error = ' --- '.join(value for value in start_error.values())
            # Initial storage states
            self._init_storage_states = init_states
            self._storages_received = dict()
            for key in self._init_storage_states:
                LOGGER.debug('Received storage state for {:s}'.format(key))
                self._storages_received[key] = True
        else:
            self._init_storage_states = None
            self._storages_received = None
        self._dispatch_sent = False

        if self.initialization_error is not None:
            LOGGER.error(self.initialization_error)

        self._horizon = parse_duration(dispatch_horizon).total_seconds()
        self._timestep = parse_duration(dispatch_timestep).total_seconds()

        self._dispatch_topic = dispatch_topic
        self._type = resource_type
        self._result_topic = '.'.join([self._dispatch_topic, self._type])

        self._init_time = None

        if isinstance(scenario, str):
            with open(scenario, 'r') as JSON:
                scenario = json.load(JSON)

        if isinstance(scenario, dict):
            scenario = Network.from_json(scenario)

        self.model = self.PLANNER(
            network=scenario,
            horizon=self._num_of_timesteps,
            timestep=self._timestep / 3600,
            name=self.component_name,
            solver=dispatch_solver
        )

        # Time of commitment
        self._commitment_time = commitment_time
        # Initializing commitment lists
        self._commitment_values = dict()
        for retailer in self.model.retailer_names:
            self._commitment_values[retailer] = [None]*self.model.horizon

        # Set initial states
        for key, value in self._init_storage_states.items():
            self.model.set_param(key, 'state', value)

        LOGGER.info("Created ED planner with horizon {:d}, solver {:s}".format(self._num_of_timesteps, dispatch_solver))

        # List listened topics to LOGGER
        LOGGER.info(f'Listening to messages from topic {self._other_topics}.')
        # LOGGER.info(f'Expecting messages with parameters {self.model.network.topics()}.')

        # to make things wait until epoch is processed
        # self._event = asyncio.Event()

    @property
    def _num_of_timesteps(self):
        return int(self._horizon // self._timestep)

    def get_time_index(self, to_string=True, fp=False):
        """ Returns time index of current economic dispatch as list of iso format strings.
        Indices are in string format if to_string is True. If fp is True function returns edges of all intervals
        instead of just interval start indices (one more).
        """
        if self._init_time is None:
            return None
        
        td = timedelta(seconds=self._timestep)
        current = parse_datetime(self._init_time)

        index = []
        N = self._num_of_timesteps+1 if fp else self._num_of_timesteps 
        for _ in range(0, N, 1):
            ind = to_iso_format_datetime_string(current) if to_string else current
            index.append(ind)
            current = current + td
        return index

    def _get_day_ahead_index(self):
        """
        Returns list of indices from 00:00 to 00:00
        """

        utc_obj = to_utc_datetime_object(self._init_time)
        td = timedelta(days=1)
        # Start time next day
        start_time = utc_obj + td
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        # End time next day
        end_time = start_time + td

        indices = []
        index_counter = 0

        while utc_obj < start_time:
            utc_obj = utc_obj + timedelta(seconds=self._timestep)
            index_counter = index_counter + 1
        while utc_obj < end_time:
            utc_obj = utc_obj + timedelta(seconds=self._timestep)
            indices.append(index_counter)
            index_counter = index_counter + 1

        return indices

    def _get_day_ahead_index_for_epoch_1(self) -> List[int]:
        """
        Returns list of indices up to end of day or end of next day.
        """

        hours_mins = self._commitment_time.split(':')

        # End of day
        utc_obj = to_utc_datetime_object(self._init_time)
        commit_time_same_day = utc_obj.replace(hour=int(hours_mins[0]), minute=int(hours_mins[1]))
        if utc_obj < commit_time_same_day:
            end_of_day = utc_obj + timedelta(days=1)
            end_of_day = end_of_day.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            end_of_day = utc_obj + timedelta(days=2)
            end_of_day = end_of_day.replace(hour=0, minute=0, second=0, microsecond=0)

        indices = []
        index_counter = 0

        while utc_obj < end_of_day:
            utc_obj = utc_obj + timedelta(seconds=self._timestep)
            indices.append(index_counter)
            index_counter = index_counter + 1

        return indices

    def _set_commitments_to_model(self):
        """
        Set day-ahead buy commitments to model.
        """

        for retailer in self._commitment_values.keys():
            LOGGER.debug("Setting commitments: " + str(self._init_time) + " " + " " + str(retailer) + ": " +
                         str(dict(zip(self.model.time_index, self._commitment_values[retailer]))) +
                         " " + str(len(self._commitment_values[retailer])))

            self.model.set_param(retailer, "commitments", dict(zip(self.model.time_index,
                                                                   self._commitment_values[retailer])))

    def _check_time_index_range(self, index, comparison_index=None):
        """
        Checks if index covers all of time index range
        """

        if comparison_index is None:
            comparison_index = self.get_time_index(to_string=False, fp=True)
        
        if index[0] <= comparison_index[0] and index[-1] >= comparison_index[-1]:
            return True
        return False
    
    def _check_time_index_interval(self, index):
        """
        Checks if the intervals in index are regular. Returns interval if regular, otherwise None
        """

        index = [parse_datetime(dt) for dt in index]
        interval = index[1] - index[0]

        for i in range(1, len(index)):
            ts = index[i] - index[i-1]
            if not ts == interval:
                return None
        return interval
        
    def _get_minimal_cover(self, index, comparison_index=None):
        """
        Get the slice of index entries that are needed to cover comparison index.
        """

        if comparison_index is None:
            comparison_index = self.get_time_index(to_string=False, fp=True)

        N = len(index)
        last_idx = next(i for i in range(1, N) if self._check_time_index_range(index[:i+1], comparison_index))
        first_idx = next(i for i in range(last_idx-1, -1, -1) if self._check_time_index_range(index[i:last_idx+1],
                                                                                              comparison_index))
        return slice(first_idx, last_idx+1)

    def _get_cover_weights(self, index, target_interval):
        """
        Gets the slice of index intervals that cover target_interval
        and weights that tell how much each interval of that slice covers of target_iinterval.
        """

        cover = self._get_minimal_cover(index, target_interval)
        ts = (target_interval[1] - target_interval[0]).total_seconds()
        index = index[cover]

        weights = []
        for k in range(len(index)-1):
            if k == 0:
                td_in = (index[k+1] - target_interval[0]).total_seconds()
            else:
                td_in  = (index[k+1] - index[k]).total_seconds()
            
            if index[k+1] > target_interval[1]:
                td_in = td_in - (index[k+1] - target_interval[1]).total_seconds()

            weights.append(td_in / ts)

        return cover, weights
    
    def _process_time_series(self, series, index):
        """
        Checks index and transforms series to using ED time index.
        """

        # check interval regularity, get interval timedelta
        interval = self._check_time_index_interval(index)
        if interval is None:
            LOGGER.debug("Time index interval is not regular.")
            return None

        index = [parse_datetime(dt) for dt in index]
        index.append(index[-1]+interval)
        ti = self.get_time_index(to_string=False, fp=True)
        
        if not self._check_time_index_range(index, ti):
            LOGGER.debug("Series: " + str(series))
            LOGGER.debug("Index: " + str(index))
            LOGGER.debug("TI : " + str(ti))
            LOGGER.debug("Time index does not cover ED index range")
            return None

        ser = []
        for k in range(self._num_of_timesteps):
            cover, weights = self._get_cover_weights(index, ti[k:k+2])
            s = [w*val for w, val in zip(weights, series[cover])]
            ser.append(sum(s))
        
        return ser

    def clear_epoch_variables(self) -> None:
        """
        Clears all the variables that are used to store information about the received input within the
        current epoch. This method is called automatically after receiving an epoch message for a new epoch.
        """

        self.model.clear()
        LOGGER.info("{:s}: Input parameters cleared for epoch {:d}".format(self.model.name,
                                                                           self._latest_epoch_message.epoch_number))

        # Setting storage states to what was received on previous epoch (or in initialization)
        if self._init_storage_states:
            for key, value in self._init_storage_states.items():
                try:
                    self.model.set_param(key, 'state', value)
                except KeyError:
                    pass  # No big deal, not our resource
                except ValueError as e:
                    LOGGER.debug("Setting state for {:s} failed due to value error.".format(key))
                    raise e
                else:
                    LOGGER.info("Set state of charge for unit {:s} on epoch {:d}".
                                format(key, self._latest_epoch_message.epoch_number))

        if self._storages_received:
            for key in self._storages_received.keys():
                self._storages_received[key] = False

        # Reset dispatch check
        self._dispatch_sent = False
        
        # set time index first entry
        self._init_time = self._latest_epoch_message.start_time

        # Removing first value from commitments list
        for retailer in self._commitment_values.keys():
            self._commitment_values[retailer].pop(0)
            self._commitment_values[retailer].append(None)

        # set buy commitments
        self._set_commitments_to_model()

    # def _check_time_inde_strict(self, index):
    #     # TODO: check that all in index minimal cover in ED time index
    #     return True

    async def process_resource_forecast(self, message_object: ResourceForecastPowerMessage,
                                        message_routing_key: str) -> None:
        """
        Handles the processing of 'resource forecasts.
        """

        index = message_object.forecast.time_index
        # processing epoch will eventually raise error if series is None if this was a message that ED expected
        series = self._process_time_series(message_object.forecast.series["RealPower"].values, index)
        try:
            self.model.set_param(message_object.resource_id, 'forecast', series)
        except KeyError:
            pass  # No big deal, not our resource
        except ValueError as e:
            LOGGER.debug("Setting resource forecast for {:s} failed due to value error.".
                         format(message_object.resource_id))
            raise e
        else:
            LOGGER.info("Set resource forecast for unit {:s}, barrier state is {!s}".
                        format(message_object.resource_id, self.model.ready_to_solve()))

    async def process_prices(self, message_object: PriceForecastStateMessage,
                             message_routing_key: str) -> None:
        """
        Handles the processing of price forecasts.
        """

        failure = False

        if message_object.prices.series["Price"].unit_of_measure == "{EUR}/(MW.h)":
            val_list = [float(value) / 1000.0 for value in message_object.prices.series["Price"].values]
        elif message_object.prices.series["Price"].unit_of_measure == "{EUR}/(kW.h)":
            val_list = [float(value) for value in message_object.prices.series["Price"].values]
        else:
            failure = True
            LOGGER.debug("ED: PriceForecast Price unit of measure not identified")

        if failure:
            # processing epoch will eventually raise error due to this value if this was a message that ED expected
            val_list = None

        try:
            self.model.set_param(message_object.marketid, 'prices', val_list)
        except KeyError:
            pass # No big deal, not our resource
        except ValueError as e:
            LOGGER.debug("Setting price forecast for {:s} failed due to value error.".format(message_object.marketid))
            raise e
        else:
            LOGGER.debug("Set price forecast for unit {:s}, to {!s}".format(message_object.marketid,
                                                                            [val for val in val_list]))
            LOGGER.info("Set price forecast for unit {:s}, barrier state is {!s}".format(message_object.marketid,
                                                                                         self.model.ready_to_solve()))

    async def process_states(self, message_object: ResourceStateMessage,
                             message_routing_key: str) -> None:
        """
        Handles the processing of storage states.
        """

        self._storages_received[message_object.source_process_id] = True
        self._init_storage_states[message_object.source_process_id] = message_object.state_of_charge.value / 100.0
        LOGGER.info("Received state of charge for unit {:s}".format(message_object.source_process_id))

    def _check_commitment_time(self, time_now: datetime) -> bool:
        """
        Returns True if commitment time hours and minutes matches given datetime value.
        """

        hours_minutes = self._commitment_time.split(':')
        if time_now.hour == int(hours_minutes[0]) and time_now.minute == int(hours_minutes[1]):
            return True
        else:
            return False

    def _check_commitment_time_before(self, time_now: datetime) -> bool:
        """
        Returns True if commitment time hours and minutes before or equal to given datetime value.
        """

        hours_minutes = self._commitment_time.split(':')
        if time_now.hour <= int(hours_minutes[0]) and time_now.minute <= int(hours_minutes[1]):
            return True
        else:
            return False

    async def process_epoch(self) -> bool:
        """
        Process the epoch and do all the required calculations.
        Assumes that all the required information for processing the epoch is available.

        Returns False, if processing the current epoch was not yet possible.
        Otherwise, returns True, which indicates that the epoch processing was fully completed.
        This also indicated that the component is ready to send a Status Ready message to the Simulation Manager.

        NOTE: this method should be overwritten in any child class.
        """

        # Dispatch has not been sent on this epoch
        # (process_epoch only called if component is ready to solve)
        if not self._dispatch_sent:
            LOGGER.info("Solving dispatch for epoch {:d}".format(self._latest_epoch))

            try:

                r = self.model.solve()

                LOGGER.info("Solved {:s} with status {:s} and termination condition {:s}".format(
                    self.model.name, r.solver.status, r.solver.termination_condition))

                results = self.model.results()

                # Set commitments from results at commitment time
                if self._commitment_time is not None and \
                        self._latest_epoch != 1 and \
                        self._check_commitment_time(to_utc_datetime_object(self._init_time)):
                    LOGGER.debug("Commit time: Day ahead indices: " + str(self._get_day_ahead_index()))

                    # Commit indices
                    indices = self._get_day_ahead_index()
                    for retailer in self._commitment_values.keys():
                        for index in indices:
                            if index >= len(self._commitment_values[retailer]):
                                break
                            self._commitment_values[retailer][index] = results[retailer][index]

                    # Set commitments (for calculation of Offers)
                    self._set_commitments_to_model()

                # Epoch 1 initialisation for commitments
                if self._commitment_time is not None and self._latest_epoch == 1:
                    LOGGER.debug("Epoch 1: Day ahead indices: " + str(self._get_day_ahead_index_for_epoch_1()))
                    # Commits to end of day OR end of next day

                    # Get commit indices
                    indices = self._get_day_ahead_index_for_epoch_1()
                    for retailer in self._commitment_values.keys():
                        for index in indices:
                            if index >= len(self._commitment_values[retailer]):
                                break
                            self._commitment_values[retailer][index] = results[retailer][index]

                    # Set commitments (for calculation of Offers)
                    self._set_commitments_to_model()

                # Sending results
                self._dispatch_sent = True
                await self._send_results_message(results)

            except Exception as error:
                description = \
                    "Unable to create or send a ResourceForecastStateDispatchMessage message in epoch {:d}: {!s}".\
                    format(self._latest_epoch, error)
                LOGGER.error(description)
        else:
            return True

    async def all_messages_received_for_epoch(self) -> bool:
        """
        Returns True, if all the messages required to start the processing for the current epoch.
        Checks only that all the required information is available.
        """
        if self._dispatch_sent:
            return self._storages_received is None or all(self._storages_received.values())
        else:
            return self.model.ready_to_solve()

    async def general_message_handler(self, message_object: Union[BaseMessage, Any],
                                      message_routing_key: str) -> None:
        """
        Forwards the message handling to the appropriate function depending on the message type.
        Assumes that the messages are not of type SimulationStateMessage or EpochMessage.
        """

        # TODO: ensure time series index compatibility
        if isinstance(message_object, PriceForecastStateMessage):
            LOGGER.debug("Received {:s} message from topic {:s}".format(
                message_object.message_type, message_routing_key))
            await self.process_prices(message_object, message_routing_key)

        elif isinstance(message_object, ResourceForecastPowerMessage):
            LOGGER.debug("Received {:s} message from topic {:s}".format(
                message_object.message_type, message_routing_key))
            await self.process_resource_forecast(message_object, message_routing_key)

        elif isinstance(message_object, ResourceStateMessage):
            LOGGER.debug("Received {:s} message from topic {:s}".format(
                message_object.message_type, message_routing_key))
            await self.process_states(message_object, message_routing_key)

        else:
            LOGGER.debug("Received unknown message: {:s}".format(str(message_object)))

        if not await self.start_epoch():
            LOGGER.debug("Waiting for other required messages before finishing epoch {:d}".format(self._latest_epoch))

    async def _send_results_message(self, results):
        """
        Sends a Dispatch message.
        """

        result = self._get_result_message(results)
        await self._rabbitmq_client.send_message(self._result_topic, result.bytes())

    def _get_result_message(self, results) -> ResourceForecastStateDispatchMessage:
        """
        Creates new dispatch message and returns it.
        Returns None, if there was a problem creating the message.
        """

        dispatch = {}
        for name, values in results.items():
            series = {"RealPower": ValueArrayBlock(Values=values, UnitOfMeasure="kW")}
            dispatch[name] = TimeSeriesBlock(TimeIndex=self.get_time_index(), Series=series)

        message = ResourceForecastStateDispatchMessage(
            SimulationId=self.simulation_id,
            Type=ResourceForecastStateDispatchMessage.CLASS_MESSAGE_TYPE,
            SourceProcessId=self.component_name,
            MessageId=next(self._message_id_generator),
            EpochNumber=self._latest_epoch,
            TriggeringMessageIds=self._triggering_message_ids,
            Dispatch=dispatch
        )

        return message

