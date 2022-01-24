# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains class for unit level component StorageUnit.
"""

import pyomo.environ as pyo
from economic_dispatch.model.units.base import _Unit


def check_state(val):
    if val is None or 0.0 <= val <= 100.0:
        return True
    else:
        return False


class StorageUnit(_Unit):
    """
    Storage unit

    ...

    Attributes
    ----------
    name: str
        name of the unit
    state: float
        Initial state of charge [%/100]
    capacity: float
        Rated storage capacity [kWh]
    max_power: float
        Power output rating [kW]
    charge_rate: float
        default: 1.0
        Charge rate in percentage of rated power [%/100]
    discharge_rate: float
        default: 1.0
        Discharge rate in percentage of rated power [%/100]
    dch_efficiency: float
        default: 0.9
        Discharge efficiency [%/100]
    ch_efficiency: float
        default: 0.9
        Charge efficiency [%/100]
    self_dch: float
        default: 0.0
        Self discharge per time step [%/100]
        
    Methods
    -------
    block_rule(block: pyo.Block)
        Builds all the optimisation model components for this unit type 
        on top of the passed block.
    topics(prefix: bool=True)
        returns a list of unit's problem instance parameter names, 
        if prefix=True '{self.name}.' is added before attribute names.
    """

    PROB_INSTANCE_PARAMS = ['state']
    TIME_INDEXED_PROB_INSTANCE_PARAMS = []

    def __init__(self, name,
                 capacity,  # Rated storage capacity - required
                 max_power,  # Power output rating - required
                 charge_rate=1.0,  # Charge rate in percentage of rated power [%/100]
                 discharge_rate=1.0,  # Discharge rate in percentage of rated power [%/100]
                 dch_efficiency=0.9,  # Discharge efficiency [%/100]
                 ch_efficiency=0.9,  # Charge efficiency [%/100]
                 self_dch=0.0,  # Self discharge per time step [%/100]
                 terminal_soc_target=0.5,  # Target state of charge for final time instant [%/100]
                 terminal_soc_bound=0.4,  # Lower bound for state fo chrage at final time instant [%/100]
                 terminal_weight=None,  # Cost weight for deviation from state target at final time instant
                 ):
        """
        Parameters
        ----------
        name: str
            name of the unit
        capacity: float
            rated storage capacity [kWh]
        max_power: float
            power output rating [kW]
        charge_rate: float
            charge rate in percentage of rated power [%/100]
        discharge_rate: float
            discharge rate in percentage of rated power [%/100]
        dch_efficiency: float
            discharge efficiency [%/100]
        ch_efficiency: float
            charge efficiency [%/100]
        self_dch: float
            self discharge per time step [%/100]
        terminal_soc_bound: float
            lower bound for state fo chrage at final time instant [%/100]
        terminal_weight: float
            cost weight for deviation from state target at final time instant
        """

        super().__init__(name)

        self.state = None
        self.capacity = capacity
        self.max_power = max_power
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.dch_efficiency = dch_efficiency
        self.ch_efficiency = ch_efficiency
        self.self_dch = self_dch
        self.terminal_soc_target = terminal_soc_target
        self.terminal_soc_bound = terminal_soc_bound
        self.terminal_weight = terminal_weight

    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, val):
        if check_state(val):
            self._state = val
        else:
            raise ValueError("Given value not accepted as state.")

    @property
    def state_kwh(self):
        """ Battery state of charge in kWh. """
        return self.state * self.capacity

    @state_kwh.setter
    def state_kwh(self, val):
        """ Set battery state of charge in kWh. """
        self.state = val / self.capacity

    def block_rule(self, block):
        """
        Builds all the optimisation model components for this unit type
        on top of the passed block.

        The underlying model should have its time index set at attr T

        Parameters
        ----------
        block: pyo.Block
            unit level block of the model.
        """

        model = block.model()
        N = model.T.last()  # last time index

        # Load variable with explicit lower and upper bounds
        # Linear bounds then applied to Power variable
        # Power variable to State transition
        def power_bounds(b, i):
            return -1 * self.discharge_rate * self.max_power, self.charge_rate * self.max_power
        block.load = pyo.Var(model.T, bounds=power_bounds)

        # Piecewise power constraints (load->power / external effect->internal effect)
        def f(x):
            if x >= 0:
                return self.ch_efficiency * x * model.timestep
            else:
                return x * model.timestep / self.dch_efficiency

        pw_pts = {}
        for idx in block.load.index_set():
            pw_pts[idx] = [block.load[idx].lb, 0, block.load[idx].ub]

        # Index set for interval edges (1:n) of the pw model
        n = len(pw_pts[0])
        block.pw_pts_idx = pyo.RangeSet(1, n)

        # Weight variables form an SOS2 set
        block.weight = pyo.Var(model.T, block.pw_pts_idx, domain=pyo.NonNegativeReals)

        # Model the SOS2 set

        # 0-1 SOS1 variable (only one is 1) for each pair of adjacent weights
        block.sos1_idx = pyo.RangeSet(1, n - 1)
        block.sos1_var = pyo.Var(model.T, block.sos1_idx, domain=pyo.Binary)

        def sos1_rule(b, i):
            return sum(b.sos1_var[i, idx] for idx in b.sos1_idx) == 1
        block.sos1_con = pyo.Constraint(model.T, rule=sos1_rule)

        # only the chosen pair of adjacent weights get 1 on the right side
        def sos12_rule(b, i, j):
            if j == 1:
                return b.weight[i, j] <= b.sos1_var[i, j]
            elif j == n:
                return b.weight[i, j] <= b.sos1_var[i, j - 1]
            else:
                return b.weight[i, j] <= b.sos1_var[i, j - 1] + b.sos1_var[i, j]
        block.sos12_relation = pyo.Constraint(model.T, block.pw_pts_idx, rule=sos12_rule)

        # Piecewise power model (x->y / load->power)
        def weight_rule(b, i):
            return sum(b.weight[i, idx] for idx in b.pw_pts_idx) == 1
        block.weight_con = pyo.Constraint(model.T, rule=weight_rule)

        def xj_init(b, i, j):
            return pw_pts[i][j - 1]
        block.xj = pyo.Param(model.T, block.pw_pts_idx, initialize=xj_init)

        def yj_init(b, i, j):
            return f(b.xj[i, j])
        block.yj = pyo.Param(model.T, block.pw_pts_idx, initialize=yj_init)

        def x_rule(b, i):
            return sum(b.weight[i, j] * b.xj[i, j] for j in b.pw_pts_idx) == b.load[i]
        block.pw_x_con = pyo.Constraint(model.T, rule=x_rule)

        block.power = pyo.Var(model.T)

        def power_rule(b, i):
            return sum(b.weight[i, j] * b.yj[i, j] for j in b.pw_pts_idx) == b.power[i]
        block.pw_power_con = pyo.Constraint(model.T, rule=power_rule)

        # State variable
        def state_bounds(b, i):
            return 0, self.capacity
        block.state = pyo.Var(model.T, bounds=state_bounds)

        def init_state_rule(b):
            return self.state_kwh
        block.init_state = pyo.Param(initialize=init_state_rule, within=pyo.Reals, mutable=True)

        # State equation [kWh]
        def state_rule(b, i):
            if i == model.T.first():
                return b.state[i] == b.init_state
            else:
                return b.state[i] == b.state[i - 1] * (1 - self.self_dch) + b.power[i - 1]
        block.state_eq = pyo.Constraint(model.T, rule=state_rule)

        # Terminal conditions
        # (N is the terminal index, i.e. last index in model.T set)

        # Lower bound
        # No terminal_weight given
        #       => terminal_soc_bound used as explicit lower bound
        # terminal_weight given
        #       => deviation of terminal_soc_bound from terminal_soc_target down
        def terminal_lb_rule(b):
            if self.terminal_weight is None:
                return b.state[N] >= self.terminal_soc_bound * self.capacity
            else:
                return b.state[N] >= (self.terminal_soc_target - self.terminal_soc_bound) * self.capacity
        block.terminal_lb = pyo.Constraint(rule=terminal_lb_rule)

        # Upper bound
        # No terminal_weight given
        #       => skip (i.e. state has upper bound already)
        # terminal_weight given
        #       => deviation of terminal_soc_bound from terminal_soc_target up
        def terminal_ub_rule(b):
            if self.terminal_weight is None:
                return pyo.Constraint.Skip
            else:
                return b.state[N] <= (self.terminal_soc_target + self.terminal_soc_bound) * self.capacity
        block.terminal_ub = pyo.Constraint(rule=terminal_ub_rule)

        if self.terminal_weight is not None:
            # ---------------------------------------------------------------------------------------------
            # indicator value is 1 if b.state[n-1] <= self.terminal_soc_target * self.capacity
            block.deviation_dir_indicator = pyo.Var(domain=pyo.Binary)

            def indicator_con1_rule(b):
                return b.state[N] - self.terminal_soc_target * self.capacity <= \
                       (1.0 - self.terminal_soc_target) * self.capacity * (1 - b.deviation_dir_indicator)
            block.indicator_con1 = pyo.Constraint(rule=indicator_con1_rule)

            def indicator_con2_rule(b):
                return b.state[N] + (self.terminal_soc_target * self.capacity - 1e-9) * b.deviation_dir_indicator \
                       >= self.terminal_soc_target * self.capacity + 1e-9
            block.indicator_con2 = pyo.Constraint(rule=indicator_con2_rule)

            # Linearize product of binary and continuous variables b.state[N] * b.deviation_dir_indicator
            # (= b.aux_prod) in terminal deviation
            block.aux_prod = pyo.Var(domain=pyo.Reals)

            # if indicator is 0 this sets the stricter product upper bound (vs con3)
            def prod_lin_con1_rule(b):
                return b.aux_prod <= self.capacity * b.deviation_dir_indicator
            block.prod_lin_con1 = pyo.Constraint(rule=prod_lin_con1_rule)

            # if indicator is 0 this sets the stricter product lower bound (vs con4)
            def prod_lin_con2_rule(b):
                return b.aux_prod >= 0.0
            block.prod_lin_con2 = pyo.Constraint(rule=prod_lin_con2_rule)

            # if indicator is 1 this sets the stricter product upper bound (vs con1)
            def prod_lin_con3_rule(b):
                return b.aux_prod <= b.state[N]
            block.prod_lin_con3 = pyo.Constraint(rule=prod_lin_con3_rule)

            # if indicator is 1 this sets the stricter product lower bound (vs con2)
            def prod_lin_con4_rule(b):
                return b.aux_prod <= b.state[N] + self.capacity * b.deviation_dir_indicator - self.capacity
            block.prod_lin_con4 = pyo.Constraint(rule=prod_lin_con4_rule)

            def terminal_dev_rule(b):
                if self.terminal_weight is None:
                    return 0.0
                else:
                    return (self.terminal_soc_target * self.capacity) * (1 - 2*b.deviation_dir_indicator) - \
                           (b.state[N] - 2 * b.aux_prod)
            block.terminal_deviation = pyo.Expression(rule=terminal_dev_rule)
            # -----------------------------------------------------------------------------------------------

        def rp_rule(b, i):
            return -1 * b.load[i]
        block.real_power = pyo.Expression(model.T, rule=rp_rule)

        def dispatch_rule(b, i):
            return -1 * b.load[i]
        block.dispatch = pyo.Expression(model.T, rule=dispatch_rule)

        def operational_cost_rule(b, i):
            return 0.0
        block.operational_cost = pyo.Expression(model.T, rule=operational_cost_rule)

        def cost_rule(b, i):
            if i == N and self.terminal_weight is not None:
                return self.terminal_weight * b.terminal_deviation + b.operational_cost[i]
            else:
                return b.operational_cost[i]
        block.cost = pyo.Expression(model.T, rule=cost_rule)

