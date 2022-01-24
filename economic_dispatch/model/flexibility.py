# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for FlexibilityRequest, LFMMarketResult and LFMMarketResults.
"""

from typing import List

import pyomo.environ as pyo

UPREGULATION = "upregulation"
DOWNREGULATION = "downregulation"


class FlexibilityRequest:
    """
    Class for creation of FlexibilityRequest model block. For stage 2 of EconomicDispatchFlexPlanner
    this block finds feasible offers among the requested amounts.
    """

    def __init__(self, resource_ids: List[str], storage_ids: List[str], direction: str, real_power_min: float,
                 real_power_req: float, bid_resolution: float, timesteps: List[bool]):

        self.resource_ids = resource_ids
        self.storage_ids = storage_ids
        self.direction = direction
        self.real_power_min = real_power_min
        self.real_power_req = real_power_req
        self.bid_resolution = bid_resolution
        self.timesteps = timesteps

    @property
    def offers(self):
        """
        Returns list from real_power_min to real_power_req iterated with bid_resolution
        """
        rp, n = self.real_power_min, 0
        offers = []
        while rp <= self.real_power_req:
            offers.append(rp)
            rp = self.real_power_min + n * self.bid_resolution
            n += 1
        return offers

    @property
    def num_offers(self):
        return len(self.offers)

    def block_rule(self, block):
        model = block.model()

        # Time indices for regulation
        def Tr_init(b):
            return [t for t in model.T if self.timesteps[t]]
        block.Tr = pyo.Set(initialize=Tr_init)

        def sign_init(b):
            return 1 if self.direction == DOWNREGULATION else -1
        block.sign = pyo.Param(initialize=sign_init)

        # Units to be regulated
        def C_init(b):
            return self.storage_ids
        block.C = pyo.Set(initialize=C_init)

        # Real power references
        def rp_ref_rule(b, t):
            return sum(model.rp_ref[t, u] for u in b.C)
        block.rp_ref = pyo.Expression(model.T, rule=rp_ref_rule)

        # Offers
        def offers_rule(b, i):
            return self.offers[i]
        block.offer = pyo.Param(model.O, rule=offers_rule)

        # Binary variable that will enforce offer feasibility if value 1
        block.o_feasible = pyo.Var(model.O, domain=pyo.Binary)

        # Create model copies that may be constrained to answer request
        def model_o_rule(b, i):
            parent = b.parent_block()
            model2 = b.model()

            # Total real power of customers in customer_ids.
            # The network copy is utilized here so we may output different amount to each offer.
            # Customer real power over all time instants
            def customers_rp_rule(blk, t):
                real_power = 0.0
                for bus in model2.Network[i].B:
                    for unit in model2.Network[i].Buses[bus].U:
                        if unit in parent.C:
                            real_power += model2.Network[i].Buses[bus].Units[unit].real_power[t]
                return real_power
            b.customers_rp = pyo.Expression(model2.T, rule=customers_rp_rule)

            # Deviation from reference
            # Regulated power over all time instants
            # Always positive
            def regulation_rule(blk, t):
                return (parent.rp_ref[t] - blk.customers_rp[t]) * parent.sign
            b.regulation = pyo.Expression(model2.T, rule=regulation_rule)
            # Downregulation -> regulation = reference rp - realized rp
            #                   offer - regulation <= 0
            # i.e. realized rp lowered at least offer (>0) amount
            # Upregulation -> regulation = -( reference rp - realized rp ) = realized rp - reference rp
            #                   offer - regulation <= 0
            # i.e. realized rp raised at least offer (>0) amount

            # offer amount each time step
            # Offer calculated only at time instants related to offer
            def offer_regulation_rule(blk, t_offer):
                if self.timesteps[t_offer]:
                    return parent.offer[i]
                else:
                    return 0.0
            b.offer_regulation = pyo.Expression(model2.T, rule=offer_regulation_rule)

            M = 10000  # TODO: replace with bounds for regulation difference or make sure large enough

            # Feasibility stuff:
            # parent.o_feasible[i] = 1
            #       -> (b.offer_regulation - parent.regulation) <= 0
            # parent.o_feasible[i] = 0
            #       -> (b.offer_regulation - parent.regulation) <= M
            # For timesteps in offer:
            def condition1_rule(blk, t):
                if self.timesteps[t]:
                    return blk.offer_regulation[t] - blk.regulation[t] + M * parent.o_feasible[i] <= M
                else:
                    return pyo.Constraint.Skip
            b.condition1 = pyo.Constraint(model2.T, rule=condition1_rule)

            return b

        block.model_o = pyo.Block(model.O, rule=model_o_rule)

        # Difference between stage 1 solution cost and new cost with regulation
        def offer_price_rule(b, i):
            return sum(model.Network[i].op_cost[t] for t in model.T) - model.price_ref
        block.offer_price = pyo.Expression(model.O, rule=offer_price_rule)

        def cost_rule(b):
            return sum(-100 * b.o_feasible[o] + b.offer_price[o] for o in model.O)
        block.cost = pyo.Expression(rule=cost_rule)


class AcceptedOffer:
    """
    Class for creation of AcceptedOffer model blocks. For stage 1 of EconomicDispatchFlexPlanner
    these blocks set real power constraints based on what regulation offers have been accepted.
    """

    def __init__(self, congestion_id: str, resource_ids: List[str], storage_ids: List[str],
                 direction: str, real_power: List[float],
                 price: float, timesteps: List[bool], real_power_ref: List[float]):

        self.congestion_id = congestion_id
        self.storage_ids = storage_ids
        self.resource_ids = resource_ids
        self.direction = direction
        self.real_power = real_power
        self.price = price
        self.timesteps = timesteps
        self.real_power_ref = real_power_ref

    def block_rule(self, block, feasibility_fulfilled=True):

        model = block.model()

        # Regulated timesteps
        def Tr_init(b):
            return [t for t in model.T if self.timesteps[t]]
        block.Tr = pyo.Set(initialize=Tr_init)

        def sign_init(b):
            return 1 if self.direction == DOWNREGULATION else -1
        block.sign = pyo.Param(initialize=sign_init)

        # Customer id set
        def C_init(b):
            return self.storage_ids
        block.C = pyo.Set(initialize=C_init)

        def customers_rp_rule(b, t):
            real_power = 0.0
            for bus in model.Network.B:
                for unit in model.Network.Buses[bus].U:
                    if unit in b.C:
                        real_power += model.Network.Buses[bus].Units[unit].real_power[t]
            return real_power
        block.customers_rp = pyo.Expression(block.Tr, rule=customers_rp_rule)

        def rp_ref_rule(b, t):
            return self.real_power_ref[t]
        block.rp_ref = pyo.Param(block.Tr, rule=rp_ref_rule)

        # Deviation from reference
        def regulation_rule(b, t):
            return (b.rp_ref[t] - b.customers_rp[t]) * b.sign
        block.regulation = pyo.Expression(block.Tr, rule=regulation_rule)

        def offer_regulation_rule(b, t):
            return self.real_power[t]
        block.offer_regulation = pyo.Expression(block.Tr, rule=offer_regulation_rule)

        if not feasibility_fulfilled:
            def lowering_bounds(b, t):
                return 0, b.offer_regulation[t]
            block.lowering_offer = pyo.Var(block.Tr, bounds=lowering_bounds)

            def lowering_cost_calculation(b):
                return sum(100000 * b.lowering_offer[t] for t in block.Tr)
            block.lowering_cost = pyo.Expression(rule=lowering_cost_calculation)

            def condition_rule(b, t):
                return b.offer_regulation[t] - b.lowering_offer[t] <= b.regulation[t]
            block.condition = pyo.Constraint(block.Tr, rule=condition_rule)

        else:
            def condition_rule(b, t):
                # do at least as much as expected
                return b.offer_regulation[t] <= b.regulation[t]
            block.condition = pyo.Constraint(block.Tr, rule=condition_rule)


class OpenOffer:
    """
    Class for creation of OpenOffer model blocks. For stage 2 of EconomicDispatchFlexPlanner
    these blocks set real power constraints based on what has been already offered so that
    no conflicts arise when multiple offers get accepted later. 
    
    Differs from Accepted order in that this sets constraints for every copy of the underlying network model
    in the stage 2 problem.
    """

    def __init__(self, congestion_id: str, resource_ids: List[str], storage_ids: List[str],
                 direction: str, real_power: List[float],
                 price: float, timesteps: List[bool], real_power_ref: List[float]):

        self.congestion_id = congestion_id
        self.resource_ids = resource_ids
        self.storage_ids = storage_ids
        self.direction = direction
        self.real_power = real_power
        self.price = price
        self.timesteps = timesteps
        self.real_power_ref = real_power_ref

    def block_rule(self, block):

        model = block.model()

        def Tr_init(b):
            return [t for t in model.T if self.timesteps[t]]
        block.Tr = pyo.Set(initialize=Tr_init)

        def sign_init(b):
            return 1 if self.direction == DOWNREGULATION else -1
        block.sign = pyo.Param(initialize=sign_init)

        # Customer id set
        def C_init(b):
            return self.storage_ids
        block.C = pyo.Set(initialize=C_init)

        def customers_rp_rule(b, t, j):
            real_power = 0.0
            for bus in model.Network[j].B:
                for unit in model.Network[j].Buses[bus].U:
                    if unit in b.C:
                        real_power += model.Network[j].Buses[bus].Units[unit].real_power[t]
            return real_power
        block.customers_rp = pyo.Expression(block.Tr, model.O, rule=customers_rp_rule)

        def rp_ref_rule(b, t):
            return self.real_power_ref[t]
        block.rp_ref = pyo.Param(block.Tr, rule=rp_ref_rule)

        # Deviation from reference
        def regulation_rule(b, t, j):
            return (b.rp_ref[t] - b.customers_rp[t, j]) * b.sign
        block.regulation = pyo.Expression(block.Tr, model.O, rule=regulation_rule)

        def offer_regulation_rule(b, t):
            return self.real_power[t]
        block.offer_regulation = pyo.Expression(block.Tr, rule=offer_regulation_rule)

        def condition_rule(b, t, j):
            return b.offer_regulation[t] <= b.regulation[t, j]
        block.condition = pyo.Constraint(block.Tr, model.O, rule=condition_rule)
