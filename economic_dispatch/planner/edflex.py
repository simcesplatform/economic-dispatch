# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains classes for EconomicDispatchFlexPlanner.
"""

from typing import List, Dict

from itertools import chain, combinations, product

import pyomo.environ as pyo

from economic_dispatch.planner.exceptions import ModelBuildError
from economic_dispatch.planner.ed import EconomicDispatchPlanner, SOLVED_STATUS
from economic_dispatch.model.network import Network
from economic_dispatch.model.flexibility import FlexibilityRequest, AcceptedOffer, OpenOffer

from pyomo.opt import TerminationCondition

# Price of offer = Price of regulation * PRICE_MULTIPLIER
PRICE_MULTIPLIER = 1.25
# Minimum price for offer
PRICE_MIN = 1.0


# from https://docs.python.org/3/library/itertools.html#itertools-recipes
# alternatively import from more-itertools package
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class EconomicDispatchFlexPlanner(EconomicDispatchPlanner):
    """
    Planner for economic dispatch and flexibility request. 
    
    Flexibility requests refer to a
    solved ED model which will be called stage 1 model, stage 2 model is build for generating offers
    for the flexibility request. 

    Attributes
    ----------
    name: str
        name of the planner
    horizon: int
        num of timesteps in model
    time_index: List[int]
        range from 0 to horizon-1
    timestep: float
        timestep length in hours
    network: Network
    model_stage1: pyo.ConcreteModel
        pyomo optimisation model object for stage 1 (dispatch)
    solve_status1: str
    model_stage2: pyo.ConcreteModel
        pyomo optimisation model object for stage 2 (flexibility request)
    solve_status2: str
    customer_map: Dict
        dict that maps resource_id (key) to customer id (value)

    Methods
    -------
    set_param(resource_id, param_name, val)
        sets parameter value for resource
    set_request(..)
        set request for stage 2
    add_lfm_result(..)
        add accepted offer to stage 1
    add_open_offer(..)
        add open offer to stage 2
    ready_to_solve(level) -> bool
        is model stage=level ready to be built and solved
    solve(level)
        solve model
    results(level)
        returns results in dict for solved model
    clear()
        clears model parameters, need to be set again

    Class Methods
    -------------
    from_json(ed_json: dict)
    """

    def __init__(self, network, customer_map=None, horizon=36, name='FlexPlanner', timestep=1.0, solver='glpk'):
        """

        :param network: Network object or dict description of network
            network containing the grid composition and units
        :param horizon: int
            optimization horizon length (def=36)
        :param name: str
            name of planner (def='Planner')
        :param timestep: float
            optimisation timestep [h]
        :param solver: str
            name of solver (def='glpk')
        """
        super().__init__(network, horizon=horizon, name=name, timestep=timestep, solver=solver)

        if customer_map is None:
            customer_map = {}
        self.accepted_offers = []
        self.flexibility_request = None
        self.open_offers = []

        self.model_stage1 = None
        self._pyo_results1 = None

        self.model_stage2 = None
        self._pyo_results2 = None

        self.customer_map = customer_map

    def update_customer_ids(self, customer_map: Dict[str, str]):
        """ Update resource id to customer id map. """
        self.customer_map.update(customer_map)

    def set_request(self, direction: str, customer_ids: List[str], real_power_min: float,
                    real_power_req: float, bid_resolution: float, timesteps: List[bool]):
        """ Sets flexibility request for model stage 2. """

        customers = [k for k, v in self.customer_map.items() if v in customer_ids]
        self.flexibility_request = FlexibilityRequest(**{
            "resource_ids": customers,
            "storage_ids": self.network.storage_names,
            "direction": direction,
            "real_power_min": real_power_min,
            "real_power_req": real_power_req,
            "bid_resolution": bid_resolution,
            "timesteps": timesteps,
        })

    def add_lfm_result(self, congestion_id: str, direction: str, customer_ids: List[str], real_power: float,
                       price: float, timesteps: List[bool], real_power_ref: List[float]):
        """
        Adds an LFMMarketResult to the stage 1 model. If result with congestion_id already exists
        amount is added to existing result amount.
        """

        customers = [k for k, v in self.customer_map.items() if v in customer_ids]
        real_power = [real_power if t else 0.0 for t in timesteps]
        result = AcceptedOffer(**{
            "congestion_id": congestion_id,
            "resource_ids": customers,
            "storage_ids": self.network.storage_names,
            "direction": direction,
            "real_power": real_power,
            "price": price,
            "timesteps": timesteps,
            "real_power_ref": real_power_ref,
        })
        # TODO: make sure no conflicts in requested regulations
        # check customer_ids intersection for two offers, if any then offer directions need to be same
        # if any common active timesteps 

        # if result with congestion_id is already in we only modify the amounts
        for r in self.accepted_offers:
            if r.congestion_id == congestion_id:
                r.real_power = [rp + real_power[i] for i, rp in enumerate(r.real_power)]
                r.price = r.price + price  # irrelevant
                return

        self.accepted_offers.append(result)

    def add_open_offer(self, congestion_id: str, direction: str, customer_ids: List[str], real_power: float,
                       price: float, timesteps: List[bool], real_power_ref: List[float]):
        """
        Adds an LFMMarketResult to the stage 2 model. If result with congestion_id already exists
        amount is added to existing result amount.
        """

        customers = [k for k, v in self.customer_map.items() if v in customer_ids]
        real_power = [real_power if t else 0.0 for t in timesteps]
        offer = OpenOffer(**{
            "congestion_id": congestion_id,
            "resource_ids": customers,
            "storage_ids": self.network.storage_names,
            "direction": direction,
            "real_power": real_power,
            "price": price,
            "timesteps": timesteps,
            "real_power_ref": real_power_ref,
        })

        # if result with congestion_id is already in we only modify the amounts
        for r in self.open_offers:
            if r.congestion_id == congestion_id:
                r.real_power = [rp + real_power[i] for i, rp in enumerate(r.real_power)]
                r.price = r.price + price  # irrelevant
                return

        self.open_offers.append(offer)

    def ready_to_solve(self, level: int = 1):
        """ Returns True if model is ready to solve. """
        if level == 1:
            return super().ready_to_solve()  # ready with or without lfm results
        elif level == 2:
            return self.solve_status1 == SOLVED_STATUS and self.flexibility_request is not None
        return False

    def build_model_stage1(self, feasibility_fulfilled=True):
        """ Builds pyomo model object. """
        model = pyo.ConcreteModel()

        model.timestep = pyo.Param(initialize=self.timestep)
        model.T = pyo.Set(initialize=self.time_index)
        model.Network = pyo.Block(rule=self.network.block_rule)

        # LFMMarketResult index
        def R_init(m):
            return range(len(self.accepted_offers))
        model.R = pyo.Set(initialize=R_init)

        # LFMMarketResult block
        def lfm_results_rule(b, i):
            return self.accepted_offers[i].block_rule(b, feasibility_fulfilled)
        model.LFMResults = pyo.Block(model.R, rule=lfm_results_rule)

        # Stage 1 price
        def price_rule(m):
            return sum(m.Network.op_cost[i] for i in model.T)
        model.price = pyo.Expression(rule=price_rule)

        if feasibility_fulfilled:
            # Objective: Stage 1 price and auxiliary terms
            def objective(m):
                return sum(m.Network.cost[i] for i in model.T)
            model.OBJ = pyo.Objective(rule=objective)

        else:
            def objective(m):
                return sum(m.Network.cost[i] for i in model.T) + \
                       sum(m.LFMResults[j].lowering_cost for j in model.R)
            model.OBJ = pyo.Objective(rule=objective)

        self.model_stage1 = model

    @property
    def solve_status1(self):
        return self._get_status(self._pyo_results1)

    def _get_stage1_results(self):
        """ Retrieves and return dispatch results in a dictionary. """
        results = {}

        for bus_name in self.model_stage1.Network.B:
            bus = self.model_stage1.Network.Buses[bus_name]
            for unit_name in bus.C:
                unit = bus.Units[unit_name]
                values = [pyo.value(unit.dispatch[i]) for i in self.model_stage1.T]
                results[unit_name] = values

        return results

    def results_stage1(self):
        """ Return results in a dictionary. If model is not solved returns None. """
        if self.solve_status1 == SOLVED_STATUS:
            return self._get_stage1_results()
        else:
            return None

    def solve_stage1(self):
        """ Solves the model if model is ready. """

        results = None
        try:
            self.build_model_stage1()
        except:
            raise ModelBuildError("edflex: solve_stage1: Unable to build stage 1 model from data.")
        else:
            results = self._solver.solve(self.model_stage1)

            # Adding variable to lower accepted offers to ensure feasibility
            if results.solver.termination_condition == TerminationCondition.infeasible:
                try:
                    self.build_model_stage1(feasibility_fulfilled=False)
                except:
                    raise ModelBuildError("edflex: solve_stage1: Unable to build stage 1 model from data, "
                                          "feasibility_fulfilled = False")
                else:
                    results = self._solver.solve(self.model_stage1)

                    for lfm in self.model_stage1.R:
                        for tr in self.model_stage1.LFMResults[lfm].Tr:
                            val = self.model_stage1.LFMResults[lfm].lowering_offer[tr].value
                            self.model_stage1.LFMResults[lfm].lowering_offer[tr].setlb(val)
                            self.model_stage1.LFMResults[lfm].lowering_offer[tr].setub(val)

        self._pyo_results1 = results

        return results

    def clear_stage1(self):
        self._pyo_results1 = None
        self.accepted_offers = []

    def build_model_stage2(self):
        """ Builds pyomo model object. """
        model = pyo.ConcreteModel()

        model.timestep = pyo.Param(initialize=self.timestep)
        model.T = pyo.Set(initialize=self.time_index)

        # Offer index
        def O_init(m):
            return list(range(self.flexibility_request.num_offers))
        model.O = pyo.Set(initialize=O_init)

        # Copy of network for every offer
        def network_rule(b, i):
            self.network.block_rule(b)
        model.Network = pyo.Block(model.O, rule=network_rule)

        # Open offer index
        def OO_init(m):
            return range(len(self.open_offers))
        model.OO = pyo.Set(initialize=OO_init)

        # Open offer constraints: offer to current request should be feasible in the case that
        # open offers get accepted later
        def open_offer_rule(b, i):
            self.open_offers[i].block_rule(b)
        model.OpenOffer = pyo.Block(model.OO, rule=open_offer_rule)

        # Unit index
        def U_init(m):
            return self.network.unit_names
        model.U = pyo.Set(initialize=U_init)

        # Stage 1 price
        def price_ref_init(m):
            return pyo.value(self.model_stage1.price)
        model.price_ref = pyo.Param(initialize=price_ref_init)

        # Real power reference from solved stage1 model
        def rp_ref_init(m, t, u):
            for bus in self.model_stage1.Network.B:
                if u in self.model_stage1.Network.Buses[bus].U:
                    return pyo.value(self.model_stage1.Network.Buses[bus].Units[u].real_power[t])
        model.rp_ref = pyo.Param(model.T, model.U, initialize=rp_ref_init)

        # Committed to first timestep results of stage 1, set constraints accordingly
        def stage1_block_rule(b, i):
            def first_step_rule(blk, u):
                for bus in model.Network[i].B:
                    if u in model.Network[i].Buses[bus].C:
                        return model.Network[i].Buses[bus].Units[u].dispatch[0] == \
                               pyo.value(self.model_stage1.Network.Buses[bus].Units[u].dispatch[0])
                return pyo.Constraint.Skip
            b.first_step = pyo.Constraint(model.U, rule=first_step_rule)
            return b
        model.stage1 = pyo.Block(model.O, rule=stage1_block_rule)

        # Flexibility request block
        model.Flexibility = pyo.Block(rule=self.flexibility_request.block_rule)

        # Objective: price of offers and auxiliary terms (num of feasible offers)
        def objective(m):
            return m.Flexibility.cost
        model.OBJ = pyo.Objective(rule=objective)

        self.model_stage2 = model

    @property
    def solve_status2(self):
        return self._get_status(self._pyo_results2)

    def _get_stage2_results(self):
        """ Retrieves and return result offers in a dictionary. """
        offers = {}
        prices_dict = {}
        for i in self.model_stage2.O:
            key = pyo.value(self.model_stage2.Flexibility.offer[i])
            val = pyo.value(self.model_stage2.Flexibility.o_feasible[i]) == 1
            offers[key] = val
            prices_dict[key] = pyo.value(self.model_stage2.Flexibility.offer_price[i])

        # largest feasible collection
        collection = self._get_acceptable_collections(offers)[-1]

        # price calculation, worst case among all possible selections from collection
        prices = []
        for c in collection:
            prices_c = []
            P = powerset(collection)
            for selection in P:
                if c in list(selection):
                    sc_sum = sum(selection)
                    prices_c.append(prices_dict[sc_sum] * c / sc_sum)
                    # = Price of offer * size of offer / sum of all offers
            prices.append(max(max(prices_c) * PRICE_MULTIPLIER, PRICE_MIN))
            # Ensures a minimum price
        direction = self.flexibility_request.direction
        # real power customer id total
        rp_ref = [pyo.value(self.model_stage2.Flexibility.rp_ref[t]) for t in self.model_stage2.T]

        return {"offers": collection, "prices": prices, "direction": direction, "real_power_ref": rp_ref}

    def _get_acceptable_collections(self, offers: Dict[int, bool]):
        """ Gets all acceptable offer combinations when offers key is offer amount and
        value is boolean indicator of offer feasbility. """

        N = len(offers)
        # feasible offers size 1
        F = [(m,) for m, feas in offers.items() if feas]

        # acceptable tuples collected to this
        Q = [()] + F

        # first iteration over list of tuples of size 2
        G = [x + y for x, y in product(F, repeat=2) if x != y and y[0] > x[0]]

        # print(*[(s, sum(s)) for s in powerset(list(offers.keys()))], sep='\n')
        # Go through all subset sizes > 1
        for _ in range(N):
            M = []  # initialize collection of accepted tuples for this size
            for g in G:
                # check feasibility
                if all(offers.get(sum(gp)) for gp in powerset(g) if len(gp) != 0):
                    M.append(g)

            Q = Q + M  # concatenate
            G = [m + x for m, x in product(M, F) if x[0] > m[-1]]

        return Q

    def results_stage2(self):
        """ Return results in a dictionary. If model is not solved returns None. """
        if self.solve_status2 == SOLVED_STATUS:
            return self._get_stage2_results()
        else:
            return None

    def solve_stage2(self):
        """ Solves the model if model is ready. """

        results = None
        try:
            self.build_model_stage2()
        except:
            raise ModelBuildError("Unable to build stage 2 model from data.")
        else:
            results = self._solver.solve(self.model_stage2)

        self._pyo_results2 = results

        return results

    def clear_stage2(self):
        self._pyo_results2 = None
        self.flexibility_request = None
        self.open_offers = []

    def results(self, level: int = 1):
        """ Return results in a dictionary. If model is not solved returns None. """
        if level == 1:
            return self.results_stage1()
        elif level == 2:
            return self.results_stage2()
        return None

    def clear(self):
        super().clear()
        self.clear_stage1()
        self.clear_stage2()

    def solve(self, level: int = 1):
        """ Solves the model if model is ready. """
        results = None
        if level == 1:
            results = self.solve_stage1()
        elif level == 2:
            results = self.solve_stage2()
        return results

    @classmethod
    def from_json(cls, ed_json: Dict):
        ed = EconomicDispatchFlexPlanner(**ed_json)
        return ed
