# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

# Attribute name in start message
# Attribute name in ED component
# Lower bound
# Upper bound
STORAGE_ATTRIBUTES = [
    ("KwhRated", "capacity", 0.0, None),
    ("KwRated", "max_power", 0.0, None),
]

# Attribute name in start message
# Attribute name in ED component
# Default value
# Lower bound
# Upper bound
STORAGE_ATTRIBUTES_OPTIONAL = [
    ("ChargeRate", "charge_rate", 100.0, 0.0, 100.0),
    ("DischargeRate", "discharge_rate", 100.0, 0.0, 100.0),
    ("DischargeEfficiency", "dch_efficiency", 90.0, 0.0, 100.0),
    ("ChargeEfficiency", "ch_efficiency", 90.0, 0.0, 100.0),
    ("SelfDischarge", "self_dch", 0.0, 0.0, 100.0),
    ("InitialStateOfCharge", "initial_state", 0.0, 0.0, 100.0)
]

# Attribute name in start message
# Attribute name in ED component
# Default value
# Lower bound
# Upper bound
STORAGE_WEIGHTS = [
    ("TerminalSOCBound", "terminal_soc_bound", 40.0, 0.0, 100.0),
    ("TerminalSOCTarget", "terminal_soc_target", None, 0.0, 100.0),
    ("TerminalWeight", "terminal_weight", None, None, None),
]

COMPONENTS = {
    "StaticTimeSeriesResource": {
        "component_key": "staticunit",
        "attributes": [],
        "optional_attributes": [],
        "weights": [],
    },
    "PriceForecaster": {
        "component_key": "market",
        "attributes": [],
        "optional_attributes": [],
        "weights": [],
    },
    "StorageResource": {
        "component_key": "storage",
        "attributes": STORAGE_ATTRIBUTES,
        "optional_attributes": STORAGE_ATTRIBUTES_OPTIONAL,
        "weights": STORAGE_WEIGHTS,
    },
}


def get_unit_param(process_param, weights_in, key, name):
    comp = COMPONENTS[key]
    if key in process_param.keys():
        unit_param = process_param[key][name]
    else:
        unit_param = None

    component_dict_base = {
        "component_key": comp["component_key"],
        "name": name
    }

    errors = dict()

    component_param = dict()
    for pkey, attr_name, lbound, ubound in comp["attributes"]:
        if lbound is not None and unit_param[pkey] < lbound:
            errors[name + "." + attr_name] = name + ": " + pkey + ": Value under lower bound"
        if ubound is not None and unit_param[pkey] > ubound:
            errors[name + "." + attr_name] = name + ": " + pkey + ": Value over upper bound"
        component_param[attr_name] = unit_param[pkey]

    init_state = None
    component_param_optional = dict()
    for pkey, attr_name, default, lbound, ubound in comp["optional_attributes"]:
        if pkey == "InitialStateOfCharge":
            init_state = unit_param.get(pkey, default)
            if lbound is not None and init_state is not None and init_state < lbound:
                errors[name + "." + attr_name] = name + ": " + pkey + ": Value under lower bound"
            if ubound is not None and init_state is not None and init_state > ubound:
                errors[name + "." + attr_name] = name + ": " + pkey + ": Value over upper bound"
            init_state = init_state / 100.0
        else:
            component_param_optional[attr_name] = unit_param.get(pkey, default)
            if lbound is not None and component_param_optional[attr_name] < lbound:
                errors[name + "." + attr_name] = name + ": " + pkey + ": Value under lower bound"
            if ubound is not None and component_param_optional[attr_name] > ubound:
                errors[name + "." + attr_name] = name + ": " + pkey + ": Value over upper bound"

    weights = dict()
    for wkey, wname, default, lbound, ubound in comp["weights"]:
        if name in weights_in.keys() and wkey in weights_in[name].keys():
            weights[wname] = weights_in[name][wkey]
        else:
            if name in weights_in.keys():
                # TerminalSOCBound given, others not -> set these to None and ignore defaults
                weights[wname] = None
            else:
                weights[wname] = weights_in["default"][wkey]
        if weights[wname] is not None:
            if lbound is not None and weights[wname] < lbound:
                errors[name + "." + wname] = name + ": " + wkey + ": Value under lower bound"
            if ubound is not None and weights[wname] > ubound:
                errors[name + "." + wname] = name + ": " + wkey + ": Value over upper bound"

    # Cast to correct values [%] -> [%/100]
    for key in component_param_optional.keys():
        component_param_optional[key] = component_param_optional[key] / 100.0
    for key in weights.keys():
        if weights[key] is not None:
            weights[key] = weights[key] / 100.0

    return {**component_dict_base, **component_param, **component_param_optional, **weights}, init_state, errors


def load_start_parameters(start_dict, name):
    process_param = start_dict["ProcessParameters"]
    ed_param = process_param["EconomicDispatch"].get(name)

    if ed_param is None:
        return None

    # Optimisation weights:
    # Default weights if not given
    if "Weights" not in ed_param.keys():
        ed_param["Weights"] = {"default":
                                   dict(zip([val[0] for val in STORAGE_WEIGHTS], [val[2] for val in STORAGE_WEIGHTS]))}
    if "default" not in ed_param["Weights"].keys():
        ed_param["Weights"]["default"] = \
            dict(zip([val[0] for val in STORAGE_WEIGHTS], [val[2] for val in STORAGE_WEIGHTS]))
    for wkey, wname, default, _, _ in STORAGE_WEIGHTS:
        if wkey not in ed_param["Weights"]["default"].keys():
            ed_param["Weights"]["default"][wkey] = default
    weights = ed_param["Weights"]

    horizon = ed_param.get("Horizon", "PT36H")
    timestep = ed_param.get("Timestep", "PT1H")

    resources = ed_param["Resources"]
    units = []
    error_dict = dict()
    init_states = dict()
    for resource in resources:
        resource_dict, init_state, errors = get_unit_param(process_param, weights, resource[0], resource[1])
        units.append(resource_dict)
        error_dict.update(errors)
        if init_state is not None:
            init_states[resource[1]] = init_state

    scenario = {"units": units}

    return horizon, timestep, scenario, init_states, error_dict
