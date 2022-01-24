# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for a simulation platform component used to optimize economic dispatch
"""

from typing import Union, Any, Dict, List, Tuple
from datetime import timedelta
from dataclasses import asdict

from isodate import parse_datetime

from tools.tools import FullLogger
from tools.message.abstract import BaseMessage
from tools.message.status import StatusMessage
from tools.message.block import TimeSeriesBlock, ValueArrayBlock, QuantityBlock

from domain_messages.resource import ResourceStateMessage
from domain_messages.price_forecaster import PriceForecastStateMessage
from domain_messages.resource_forecast import ResourceForecastPowerMessage
from domain_messages.dispatch import ResourceForecastStateDispatchMessage
from domain_messages.InitCISCustomerInfo import InitCISCustomerInfoMessage
from domain_messages.Offer import OfferMessage
from domain_messages.Request import RequestMessage
from domain_messages.LFMMarketResult import LFMMarketResultMessage

from economic_dispatch.planner import EconomicDispatchFlexPlanner
from economic_dispatch.utils import Barrier
from economic_dispatch.simulation.economic_dispatch import EconomicDispatch
from economic_dispatch.simulation.dataclasses import OfferStorage, OfferBase, Offer, Congestion

from pyomo.opt import TerminationCondition

LOGGER = FullLogger(__name__)

# Max iterations in requests
REQ_MAX_ITER = 15.0
# min bid + REQ_MAX_ITER * bid resolution


def ad_to_timesteps(activation_time: str, duration: int, index: List[str]):
    """ Takes timing parameters activation time and duration, and index. Returns
    indicator for steps in index when the timing is active, and subset of index for those activated steps."""

    timesteps = []
    ad_index = []
    activation_time = parse_datetime(activation_time)
    stop_time = activation_time + timedelta(minutes=duration)
    for ind in index:
        time_index = parse_datetime(ind)
        activated = activation_time <= time_index < stop_time
        timesteps.append(activated)
        if activated:
            ad_index.append(ind)
    return timesteps, ad_index


ED_STAGE = 1
FLEX_REQUEST_STAGE = 2
ED_READY_STAGE = 3


class EconomicDispatchFlex(EconomicDispatch):
    """
    A simulation platform component that optimizes power production/consumption over a given horizon given the prices
    and states.
    """

    PLANNER = EconomicDispatchFlexPlanner

    def __init__(self, scenario: Union[str, Dict],
                 param_topics: List[str],
                 dispatch_horizon: str,
                 dispatch_timestep: str,
                 dispatch_solver: str,
                 dispatch_topic: str,
                 resource_type: str,
                 market_id: str,
                 offer_topic: str,
                 commitment_time: str,
                 skip_open_offers: bool = False,
                 initialization_error: str = None,
                 ):

        super().__init__(scenario=scenario,
                         param_topics=param_topics,
                         dispatch_horizon=dispatch_horizon,
                         dispatch_timestep=dispatch_timestep,
                         dispatch_solver=dispatch_solver,
                         dispatch_topic=dispatch_topic,
                         resource_type=resource_type,
                         commitment_time=commitment_time,
                         initialization_error=initialization_error)

        self._offer_topic = offer_topic

        self._market_id = market_id
        self._skip_open_offers = skip_open_offers

        # TODO: change(?) so market_id can be a list
        self._lfm_ready_barrier = Barrier([market_id])
        self._lfm_result_count = {self._market_id: 0}
        self._lfm_results_ready = {self._market_id: False}

        self._latest_congestion = None
        self._offer_storage = OfferStorage(size=self._num_of_timesteps)
        self._accepted_offers = []

        # Save requests received from LFM at start of epoch (before flexibility stage)
        self._requests_before_flex = []

        self.epoch_stage = None
        self._request_to_process = False

    @property
    def epoch_stage(self):
        return self._epoch_stage

    @epoch_stage.setter
    def epoch_stage(self, val):
        self._epoch_stage = val
        LOGGER.info("Epoch stage set to {!s}".format(self._epoch_stage))

    def clear_epoch_variables(self) -> None:
        """
        Clears all the variables that are used to store information about the received input within the
        current epoch. This method is called automatically after receiving an epoch message for a new epoch.
        """
        super().clear_epoch_variables()

        self._lfm_ready_barrier.reset()
        self._lfm_result_count[self._market_id] = 0
        self._lfm_results_ready[self._market_id] = False

        self._accepted_offers = []
        self._requests_before_flex = []
        self._request_to_process = False

        self.epoch_stage = ED_STAGE

    async def process_epoch_stage1(self) -> None:
        """
        Handles the optimization and sends the results
        """

        r = self.model.solve(level=1)
        LOGGER.info("Solved {:s} level 1 with status {:s} and termination condition {:s}".
                    format(self.model.name, r.solver.status, r.solver.termination_condition))

        results = self.model.results(level=1)
        await self._send_results_message(results)

    def _set_open_offers(self) -> None:
        # get open offers and set them for stage 2 problem
        start_time = self.get_time_index()[1]  # first step dispatch set from stage 1
        open_offers = self._offer_storage.get_open_offers(start_time, self._accepted_offers)

        time_index = self.get_time_index()
        for offer in open_offers:
            timesteps, idx = ad_to_timesteps(
                activation_time=offer.activation_time,
                duration=offer.duration,
                index=time_index
            )

            # if does not apply for our time range
            if not any(timesteps):
                continue

            # stored reference to new time index
            ref = {k: v for k, v in zip(offer.time_index, offer.real_power_reference)}
            rp_ref = [ref[t] if t in idx else None for t in time_index]
            result = {
                "congestion_id": offer.congestion_id,
                "customer_ids": offer.customer_ids,
                "direction": offer.direction,
                "real_power": offer.real_power,
                "price": offer.price,
                "timesteps": timesteps,
                "real_power_ref": rp_ref,
            }
            self.model.add_open_offer(**result)
        LOGGER.debug("Set {:d} open offers for level 2 model.".format(len(open_offers)))

    async def process_epoch_stage2(self) -> None:
        """
        Handles the optimization with flexibility request and sends the results
        """

        if not self._skip_open_offers:
            self._set_open_offers()

        build_failed = False
        try:
            r = self.model.solve(level=2)
        except:
            LOGGER.debug("ProcessEpochStage2: Model build failed on level 2: sending zero offers")
            build_failed = True

        if not build_failed:
            # Problem solved, results are available
            LOGGER.info("ProcessEpochStage2: Solved {:s} level 2 with status {:s} and termination condition {:s}".
                        format(self.model.name, r.solver.status, r.solver.termination_condition))
            if r.solver.termination_condition == TerminationCondition.infeasible:
                # Problem infeasible?, results not available
                LOGGER.info("ProcessEpochStage2: Problem not solved on level 2, sending 0 offers")
                # Send zero offers for this congestion, do not save offers to OfferStorage
                await self._send_zero_offer_message_no_request(congestion_id=self._latest_congestion.congestion_id,
                                                               activation_time=self._latest_congestion.activation_time,
                                                               duration=self._latest_congestion.duration,
                                                               direction=self._latest_congestion.direction)
                self._request_to_process = False
                return

            results = self.model.results(level=2)
            offer_results = results["offers"]
            prices = results["prices"]

            offers = []
            count = len(offer_results)
            congestion = self._latest_congestion
            offer_base = OfferBase(
                **asdict(congestion),
                time_index=self.get_time_index(),
                real_power_reference=results["real_power_ref"],
                offer_count=count,
            )

        else:
            # Problem not solved, results are not available
            LOGGER.info("ProcessEpochStage2: Problem not solved on level 2, sending 0 offers")
            # Send zero offers for this congestion, do not save offers to OfferStorage
            await self._send_zero_offer_message_no_request(congestion_id=self._latest_congestion.congestion_id,
                                                           activation_time=self._latest_congestion.activation_time,
                                                           duration=self._latest_congestion.duration,
                                                           direction=self._latest_congestion.direction)

            self._request_to_process = False
            return

        if count == 0:
            # Zero offers
            LOGGER.info("ProcessEpochStage2: Zero offers, sending 0 offers")
            # Send zero offers for this congestion, do not save offers to OfferStorage
            await self._send_zero_offer_message_no_request(congestion_id=self._latest_congestion.congestion_id,
                                                           activation_time=self._latest_congestion.activation_time,
                                                           duration=self._latest_congestion.duration,
                                                           direction=self._latest_congestion.direction)

            self._request_to_process = False
            return

        for i, (offer_result, price) in enumerate(zip(offer_results, prices), 1):
            offer_id = "{:s}-{:s}-{:d}".format(self._component_name, congestion.congestion_id, i)
            offer = Offer(
                **asdict(offer_base),
                offer_id=offer_id,
                real_power=offer_result,
                price=price,
            )

            # print(json.dumps(asdict(offer), indent=3))
            self._offer_storage.append(offer)
            offers.append(offer)

        # Send offers related to latest congestion
        await self._send_offer_messages(self._latest_congestion.congestion_id)
        self._request_to_process = False

    async def process_epoch(self) -> bool:
        """
        Handles the optimization and sends the results
        """

        try:
            if self.epoch_stage == ED_STAGE:
                # ED (LFMMarketResult in problem constr)
                LOGGER.info("ProcessEpoch: Starting ED calculations in epoch {:d}.".format(self._latest_epoch))
                await self.process_epoch_stage1()
                self.epoch_stage = FLEX_REQUEST_STAGE

                # Process requests received before flexibility stage
                for message in self._requests_before_flex:
                    # Process again only if request has not been calculated already
                    if self._offer_storage.check_congestion(message.congestion_id):
                        LOGGER.info("ProcessEpoch: Found congestion id: {cong_id}, sending zero offer".format(
                            cong_id=message.congestion_id))
                        await self._send_zero_offer_message(message)
                    else:
                        await self.process_request(message, '')
                        # ED w/ flexibility request
                        # Process if Request to be processed:
                        if self._request_to_process:
                            LOGGER.info("ProcessEpoch: Starting to process "
                                        "request (before flex) in epoch {:d}.".format(self._latest_epoch))
                            await self.process_epoch_stage2()

                # Empty requests to make sure not called again in this epoch
                self._requests_before_flex = []

            elif self.epoch_stage == FLEX_REQUEST_STAGE:
                # ED w/ flexibility request
                # Check if request already calculated
                if self._offer_storage.check_congestion(self._latest_congestion.congestion_id):
                    LOGGER.info("ProcessEpoch: Found congestion id: {cong_id}, sending zero offer".format(
                        cong_id=self._latest_congestion.congestion_id))
                    await self._send_zero_offer_message_no_request(congestion_id=self._latest_congestion.congestion_id,
                                                                   activation_time=
                                                                   self._latest_congestion.activation_time,
                                                                   duration=self._latest_congestion.duration,
                                                                   direction=self._latest_congestion.direction)
                else:
                    LOGGER.info("ProcessEpoch: Starting to process request in epoch {:d}.".format(self._latest_epoch))
                    await self.process_epoch_stage2()

        except Exception as error:
            description = "ProcessEpoch: Unable to process epoch {:d}: {!s}".format(self._latest_epoch, error)
            LOGGER.error(description)
            await self.send_error_message(description)
            return False

        if self.epoch_stage == ED_READY_STAGE:
            return self._storages_received is None or all(self._storages_received.values())
        else:
            return False

    async def process_request(self, message_object: RequestMessage, message_routing_key: str) -> None:
        """
        Process Request message.
        """

        if self.epoch_stage != FLEX_REQUEST_STAGE:
            # Received offer before flex stage
            LOGGER.debug("ProcessRequest: Received request {c_id} before flex stage".format(
                c_id=message_object.congestion_id))
            self._requests_before_flex.append(message_object)
            return

        # check if customer in our customer ids
        customer_id_list = []
        inv_customer_map = {val: key for key, val in self.model.customer_map.items()}
        for customer_id in message_object.customer_ids:
            LOGGER.debug("ProcessRequest: Checking {id1}".format(id1=customer_id))
            if customer_id in inv_customer_map.keys() and \
                    inv_customer_map[customer_id] in self.model.network.unit_names:
                LOGGER.debug("ProcessRequest: Customer {id1} found in network".format(id1=customer_id))
                customer_id_list.append(customer_id)

        if not len(customer_id_list) > 0:
            LOGGER.debug("ProcessRequest: Flexibility request customer ids not among component's customer ids.")
            await self._send_zero_offer_message(message_object)
            return

        self._latest_congestion = Congestion(
            congestion_id=message_object.congestion_id,
            customer_ids=customer_id_list,
            activation_time=message_object.activation_time,
            duration=int(message_object.duration.value),
            direction=message_object.direction
        )

        timesteps, _ = ad_to_timesteps(
            activation_time=message_object.activation_time,
            duration=int(message_object.duration.value),
            index=self.get_time_index()
        )
        LOGGER.debug("ProcessRequest: Flexibility request timesteps: {timelist}".format(timelist=timesteps))

        # if does not apply for our timerange
        if not any(timesteps):
            LOGGER.debug("ProcessRequest: Flexibility request not in component's current time range")
            await self._send_zero_offer_message(message_object)
            return

        request = {
            "customer_ids": customer_id_list,
            "direction": message_object.direction,
            "real_power_min": message_object.real_power_min.value,
            "real_power_req": min([message_object.real_power_request.value,
                                   message_object.real_power_min.value + message_object.bid_resolution.value *
                                   REQ_MAX_ITER]),
            "bid_resolution": message_object.bid_resolution.value,
            "timesteps": timesteps,
        }

        if request["real_power_req"] != message_object.real_power_request.value:
            LOGGER.debug("ProcessRequest: Modified request value to ensure iteration is not too large!!")

        # Check to see if request already handled
        if not self._offer_storage.check_congestion(message_object.congestion_id):
            self.model.set_request(**request)
        else:
            LOGGER.info("ProcessRequest: Request {cong_id} already handled".format(
                cong_id=message_object.congestion_id))
            await self._send_zero_offer_message(message_object)

        self._request_to_process = True
        LOGGER.info("ProcessRequest: Set flexibility request from {:s}".format(message_object.source_process_id))

    async def process_lfm_market_result(self, message_object: LFMMarketResultMessage, message_routing_key: str) -> None:
        """
        Process LFMMarketResult message.
        """

        market_id = message_object.source_process_id
        if market_id != self._market_id:
            return

        if message_object.result_count == 0:
            self._lfm_results_ready[market_id] = True
            LOGGER.info("ProcessLFMMarketResult: LFMMarketResult from "
                        "{:s} with count 0 processed.".format(message_object.source_process_id))
            return

        congestion_id = message_object.congestion_id
        offer_id = message_object.offer_id
        offer = self._offer_storage.get(offer_id)
        if offer is None:
            LOGGER.debug("ProcessLFMMarketResult: Offer {:s} not found".format(str(offer_id)))

            # Check still to see if all received
            # All lfm results received at start of epoch?
            self._lfm_result_count[market_id] += 1
            if self._lfm_result_count[market_id] == message_object.result_count:
                self._lfm_results_ready[market_id] = True

            return
        self._accepted_offers.append(offer_id)

        # All lfm results received at start of epoch?
        self._lfm_result_count[market_id] += 1
        if self._lfm_result_count[market_id] == message_object.result_count:
            self._lfm_results_ready[market_id] = True

        time_index = self.get_time_index()
        timesteps, idx = ad_to_timesteps(
            activation_time=message_object.activation_time,
            duration=int(message_object.duration.value),
            index=time_index
        )

        # if does not apply for our timerange
        if not any(timesteps):
            LOGGER.debug("ProcessLFMMarketResult: LFMMarketResult not in component's current time range")
            return

        rp = message_object.real_power
        if isinstance(rp, QuantityBlock):
            real_power = rp.value
        elif isinstance(rp, TimeSeriesBlock):
            # TODO: for now only constants allowed (value from first of series)
            real_power = rp.series["Regulation"].values[0]

        # stored reference to new time index
        ref = {k: v for k, v in zip(offer.time_index, offer.real_power_reference)}
        rp_ref = [ref[t] if t in idx else None for t in time_index]
        result = {
            "congestion_id": congestion_id,
            "customer_ids": message_object.customerids,
            "direction": message_object.direction,
            "real_power": real_power,
            "price": message_object.price.value,
            "timesteps": timesteps,
            "real_power_ref": rp_ref,
        }

        self.model.add_lfm_result(**result)
        LOGGER.info("ProcessLFMMarketResult: Added LFMMarketResult from {:s}".format(message_object.source_process_id))

    async def process_customer_info(self, message_object: InitCISCustomerInfoMessage, message_routing_key: str) -> None:
        """ Process CustomerInfo message. """
        customer_ids = message_object.customer_id
        LOGGER.debug("ProcessCustomerInfo: Customer ids: {cust_list}".format(cust_list=customer_ids))
        resource_ids = message_object.resource_id
        LOGGER.debug("ProcessCustomerInfo: Resource ids: {res_list}".format(res_list=resource_ids))
        self.model.update_customer_ids({r: c for c, r in zip(customer_ids, resource_ids)})
        LOGGER.debug("ProcessCustomerInfo: Customer map: {cust_map}".format(cust_map=self.model.customer_map))
        LOGGER.info("ProcessCustomerInfo: Customer ids from {:s} updated.".format(message_object.source_process_id))

    async def process_lfm_ready(self, message_object: StatusMessage, message_routing_key: str) -> None:
        """
        Process ready message from LFM.
        """

        self._lfm_ready_barrier.process_arrive(message_object.source_process_id)
        # after dispatch epoch is ready when all lfms are ready
        if self._lfm_ready_barrier.pass_state and self._dispatch_sent:
            LOGGER.info("ProcessLFMReady: All LFM ready messages have been received for epoch {:d}".format(
                self._latest_epoch))
            self.epoch_stage = ED_READY_STAGE

    async def all_messages_received_for_epoch(self) -> bool:
        """
        Returns True, if all the messages required to start the processing for the current epoch.
        Checks only that all the required information is available.
        """

        if self.epoch_stage == ED_STAGE:
            # process if all inputs received
            return self.model.ready_to_solve(level=1) and all(self._lfm_results_ready.values())

        elif self.epoch_stage == FLEX_REQUEST_STAGE:
            # process at new request
            return self._request_to_process

        elif self.epoch_stage == ED_READY_STAGE:
            # process -> check epoch ready condition
            return True

        # error, shouldn't get here anyway if stages handled correctly

    async def general_message_handler(self, message_object: Union[BaseMessage, Any],
                                      message_routing_key: str) -> None:
        """
        Forwards the message handling to the appropriate function depending on the message type.
        Assumes that the messages are not of type SimulationStateMessage or EpochMessage.
        """

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

        elif isinstance(message_object, LFMMarketResultMessage):
            LOGGER.debug("Received {:s} message from topic {:s}".format(
                message_object.message_type, message_routing_key))
            await self.process_lfm_market_result(message_object, message_routing_key)

        elif isinstance(message_object, InitCISCustomerInfoMessage):
            LOGGER.debug("Received {:s} message from topic {:s}".format(
                message_object.message_type, message_routing_key))
            await self.process_customer_info(message_object, message_routing_key)

        elif isinstance(message_object, StatusMessage):
            LOGGER.debug("Received {:s} message from topic {:s}".format(
                message_object.message_type, message_routing_key))
            await self.process_lfm_ready(message_object, message_routing_key)

        elif isinstance(message_object, RequestMessage):
            try:
                LOGGER.debug(
                    "Received {:s} message from topic {:s}".format(message_object.message_type, message_routing_key))
                await self.process_request(message_object, message_routing_key)
            except Exception:
                pass

        else:
            LOGGER.debug("Received unknown message: {:s}".format(str(message_object)))

        if isinstance(message_object, StatusMessage) and (message_object.source_process_id != self._market_id or
                                                          message_object.epoch_number == 0):
            LOGGER.debug("Ignored status message for epoch {:d} from {:s}".format(self._latest_epoch,
                                                                                  message_object.source_process_id))

        elif not await self.start_epoch():
            LOGGER.debug("Waiting for other required messages before processing epoch {:d}".format(
                self._latest_epoch))

    async def _send_results_message(self, results):
        """
        Sends a Dispatch message.
        """

        result = self._get_result_message(results)
        self._dispatch_sent = True

        LOGGER.debug(str(result))

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

    async def _send_offer_messages(self, searchCongestionId=None):
        """
        Sends a Offer message.
        """

        if searchCongestionId is None:
            offers, market_ids = self._get_offer_messages()
        else:
            offers, market_ids = self._get_offer_messages(searchCongestionId)
        for offer, market_id in zip(offers, market_ids):
            await self._rabbitmq_client.send_message(self._offer_topic + "." + market_id, offer.bytes())

    def _get_offer_messages(self, searchCongestionId=None) -> Tuple[List[OfferMessage], List[str]]:
        """
        Creates new offer message and returns it.
        Returns None, if there was a problem creating the message.
        """

        messages = []
        market_ids = []
        if searchCongestionId is None:
            # Offers stored this epoch
            offers = self._offer_storage.latest(full_step=True)
        else:
            # Offers stored for this congestion id but not accepted
            offers = self._offer_storage.get_congestion_offers(searchCongestionId, self._accepted_offers)

        for offer in offers:
            _, time_index = ad_to_timesteps(offer.activation_time, offer.duration, self.get_time_index())

            rp_list = [offer.real_power for _ in time_index]
            rp_ts_block = TimeSeriesBlock(time_index, {"Regulation": ValueArrayBlock(rp_list, "kW")})

            message = OfferMessage(
                SimulationId=self.simulation_id,
                Type=OfferMessage.CLASS_MESSAGE_TYPE,
                SourceProcessId=self.component_name,
                MessageId=next(self._message_id_generator),
                EpochNumber=self._latest_epoch,
                TriggeringMessageIds=self._triggering_message_ids,
                ActivationTime=offer.activation_time,
                Duration=offer.duration,
                Direction=offer.direction,
                RealPower=rp_ts_block,
                Price=offer.price * sum(rp_ts_block.series["Regulation"].values),
                # Price to EUR (offer.price = EUR/kWh)
                CongestionId=offer.congestion_id,
                CustomerIds=offer.customer_ids,
                OfferId=offer.offer_id,
                OfferCount=offer.offer_count,
            )

            market_id = self._market_id
            messages.append(message)
            market_ids.append(market_id)

        return messages, market_ids

    async def _send_zero_offer_message(self, message_object: RequestMessage):
        """Sends offer message with zero OfferCount and zero customers as reply to provided request"""

        message = OfferMessage(
            SimulationId=self.simulation_id,
            Type=OfferMessage.CLASS_MESSAGE_TYPE,
            SourceProcessId=self.component_name,
            MessageId=next(self._message_id_generator),
            EpochNumber=self._latest_epoch,
            TriggeringMessageIds=self._triggering_message_ids,
            ActivationTime=message_object.activation_time,
            Duration=message_object.duration,
            Direction=message_object.direction,
            RealPower=TimeSeriesBlock([message_object.activation_time], {"Regulation": ValueArrayBlock([0], "kW")}),
            Price=0.0,
            CongestionId=message_object.congestion_id,
            CustomerIds=None,
            OfferId="{:s}-{:s}-{:d}".format(self._component_name, message_object.congestion_id, 0),
            OfferCount=0
        )

        await self._rabbitmq_client.send_message(self._offer_topic + "." + self._market_id, message.bytes())

    async def _send_zero_offer_message_no_request(self, congestion_id: str,
                                                  activation_time: str,
                                                  duration: int,
                                                  direction: str = OfferMessage.ALLOWED_DIRECTION_VALUES[0]):
        """Sends offer message with zero OfferCount and zero customers as reply to provided request"""

        message = OfferMessage(
            SimulationId=self.simulation_id,
            Type=OfferMessage.CLASS_MESSAGE_TYPE,
            SourceProcessId=self.component_name,
            MessageId=next(self._message_id_generator),
            EpochNumber=self._latest_epoch,
            TriggeringMessageIds=self._triggering_message_ids,
            ActivationTime=activation_time,
            Duration=duration,
            Direction=direction,
            RealPower=TimeSeriesBlock([activation_time], {"Regulation": ValueArrayBlock([0], "kW")}),
            Price=0.0,
            CongestionId=congestion_id,
            CustomerIds=None,
            OfferId="{:s}-{:s}-{:d}".format(self._component_name, congestion_id, 0),
            OfferCount=0
        )

        await self._rabbitmq_client.send_message(self._offer_topic + "." + self._market_id, message.bytes())
