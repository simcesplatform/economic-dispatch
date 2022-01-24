# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

from typing import List, Union
from dataclasses import dataclass
from datetime import timedelta

from isodate import parse_datetime


@dataclass
class Congestion:

    congestion_id: Union[str, None]
    customer_ids: Union[List[str], None]
    activation_time: Union[str, None]
    duration: Union[int, None]
    direction: Union[str, None]


@dataclass
class Request(Congestion):

    real_power_min: float
    real_power_request: float
    bid_resolution: float


@dataclass
class OfferBase(Congestion):

    time_index: List
    real_power_reference: List
    offer_count: int


@dataclass
class Offer(OfferBase):

    offer_id: str
    real_power: float
    price: float


@dataclass
class LFMResult(Congestion):

    direction: Union[str, None]
    real_power: Union[float, None]
    price: Union[float, None]
    offer_id: Union[str, None]


class ParamStorage:
    """ Storage for objects of dataclasses above. """

    def __init__(self, size: int, id_field: str):
        self.id_field = id_field
        self.params = [[] for _ in range(size)]

    def append(self, congestion):
        self.params[-1].append(congestion)

    def step(self):
        self.params.pop(0)
        self.params.append([])

    def get(self, param_id: str):
        for param_list in self.params:
            for param in param_list:
                if getattr(param, self.id_field) == param_id:
                    return param
        return None

    def latest(self, full_step: bool=False):
        try:
            if full_step:
                return self.params[-1]
            return self.params[-1][-1]
        except IndexError:
            return None


class OfferStorage(ParamStorage):

    def __init__(self, size: int):
        super().__init__(size=size, id_field="offer_id")
    
    def get_open_offers(self, start_time: str, exclude_ids: List[str]):
        open_offers = []
        start = parse_datetime(start_time)
        for param_list in self.params:
            for param in param_list:
                activation_time = parse_datetime(param.activation_time)
                stop_time = activation_time + timedelta(minutes=param.duration)
                if stop_time > start and param.offer_id not in exclude_ids:
                    open_offers.append(param)
        return open_offers

    def get_congestion_offers(self, congestion_id: str, exclude_ids: List[str]) -> List:
        offers = []
        for param_list in self.params:
            for param in param_list:
                if congestion_id in param.offer_id and param.offer_id not in exclude_ids:
                    offers.append(param)
        return offers

    def check_congestion(self, congestion_id: str):
        for param_list in self.params:
            for param in param_list:
                if congestion_id in param.offer_id:
                    return True
        return False
