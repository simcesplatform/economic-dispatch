# economic-dispatch

ProCemPlus - Economic dispatch component

## Requirements

- python 3.7
- pip for installing requirements

Solver with LP or MILP (storage uses binary variables) capabilities is required.

See for example:
- GLPK
- CBC
- CPLEX (academic/commercial license)
- Gurobi (academic/commercial license)

If used, the Dockerfile included takes care of these.

## Model & planning

### Units

- Units in the model are based on `_Unit` class:
- Units have resource parameters which are taken at construction, and problem instance parameters
which need to be set after planner creation with planner's `set_param` method.
- Common methods for units:
    - `__init__` (constructor)
        - takes resource parameters as arguments
    - `block_rule`
        - builds pyomo Block optimisation model block for the unit
- Units have class constants
    - `PROB_INSTANCE_PARAMS`: list of attribute names which are the problem instance parameters of unit type.
    - `TIME_INDEXED_PROB_INSTANCE_PARAMS`: list of problem instance parameter names that are time indexed in the model, `set_param` method of the model will check input values are appropriate length for these.
- Unit `block_rule` should build following attributes on the passed block:
    - `real_power`: pyomo Expression over `model.T` (underlying model from `block.model()`)
    - `operational_cost`: pyomo Expression over `model.T`, cost with no auxiliary terms
    - `cost`: pyomo Expression over `model.T`, cost with potential auxiliary terms
    - `dispatch` (only for controllable units): pyomo Expression over `model.T`
- Currently implemented unit types are:
    - `StaticUnit`
        - problem instance parameter: `forecast`, power forecast
        - controllable: No
        - `component_key`: `"staticunit"`
    - `StorageUnit`
        - problem instance parameter: `state`
        - controllable: Yes
        - `component_key`: `"storage"`
    - `Retailer`
        - problem instance parameter: `prices`, price forecast
        - controllable: Yes
        - `component_key`: `"market"`
    - `Generator` (not registered for the factory)
        - problem instance parameter: -
        - controllable: Yes

### Model structure
- Unit blocks are the lowest level blocks in the model
- Unit blocks belong to `Bus` block of a `Network` block
- The structure need not be specified, if only units are provided they are considered to belong to a single point, this is enough for Economic Dispatch
- TODO: LOP...(if `Line` list is given local balance constraints (`Bus` blocks) will enforce current law, TODO: voltage law needed in some cases, etc)

### Scenario definition

Scenario can be built by giving the objects to the `Network` constructor.

```python
storage_unit = StorageUnit('Battery', 100.0, 10.0)
static_generator = StaticUnit('CHP')
retailer = Retailer('ElectricityMarket')
static_load = StaticUnit('Demand')

units = [storage_unit, static_generator, retailer, static_load]
bus = Bus(name='single-point', units=units)
network = Network(name='Network', buses=[bus], lines=[])
```

or

Scenario can be built from a dictionary which holds key-value arguments for the components and the `component_key` which is required
for the `UnitFactory` to know what type of unit is in question. See the `DEFAULT_REGISTRY` and `register_component` in [economic_dispatch/model/factory.py](economic_dispatch/model/factory.py) for more.

```python
units = {
    "units": [
        {
            "component_key": "storage",
            "name": "Battery",
            "capacity": 100.0,
            "max_power": 10.0
        },
        {
            "component_key": "staticunit",
            "name": "CHP"
        },
        {
            "component_key": "staticunit",
            "name": "Demand"
        },
        {
            "component_key": "market",
            "name": "ElectricityMarket"
        }
    ]
}
network = Network.from_json(units)
```

Test with

    python -m unittest discover -s economic_dispatch/model/tests

### Planners

`EconomicDispatchPlanner` class (defined in [economic_dispatch/planner/ed.py](economic_dispatch/planner/ed.py)) contains the optimisation model and solving related methods.
- Planner attributes:
    - `name`
    - `horizon`: int, problem horizon
    - `network`: `Network` object [economic_dispatch/model/network.py](economic_dispatch/model/network.py)
    - `model`: pyomo ConcreteModel which consists of the following:
        - `T`: pyomo Set, time index set, integers from `0` to `horizon-1`
        - `timestep`: pyomo Param, timestep in hours
        - `Network`: pyomo Block by `block_rule` of `network`
        - `OBJ`: pyomo Objective, optimisation objective which is minimized
- Planner methods:
    - `set_param`: set problem instance parameter for resource
    - `ready_to_solve`: is model ready to be built and solved (problem instance parameters have been set (after creation or after clear))
    - `solve`: solve model 
    - `results`: get results from units' dispatch block attributes if model has been solved
    - `clear`: clear problem instance, by default this is a shallow clear, meaning that parameters need to be set again but the underlying values are not cleared (`_shallow_clear` attribute)

Continuing where we left in the scenario definition example
```python
initial_state = 0.7 # %/100
forecast_gen = [5.0, 5.0, 5.0]
forecast_price = [1.0, 1.0, 1.0]
forecast_load = [-20.0, -20.0, -20.0]

ed = EconomicDispatchPlanner(network, horizon=3, solver='glpk')

ed.ready_to_solve() # will return False

ed.set_param('CHP', 'forecast', forecast_gen)
ed.set_param('Battery', 'state', initial_state)
ed.set_param('ElectricityMarket', 'prices', forecast_price)
ed.set_param('Demand', 'forecast', forecast_load)

ed.ready_to_solve() # will now return True

ed.solve()
results = ed.results()
#print(json.dumps(results, indent=4))

ed.clear()
ed.ready_to_solve() # will now return False
```

`EconomicDispatchFlexPlanner` class (defined in [economic_dispatch/planner/edflex.py](economic_dispatch/planner/edflex.py))
- A two-stage planner where the first stage solves ED problem and second problem calculates based on a flexibility request what kind of regulation is feasible given the stage 1 result.
- Planner attributes based on `EconomicDispatchPlanner`, additionally:
    - `model_stage1`: pyomo ConcreteModel for stage 1
    - `model_stage2`: pyomo ConcreteModel for stage 2
    - `customer_map`: dict with key resource_id and value customer_id, used to target set offers and requests to certain resources
- Planner methods `solve` and `results` now take stage as `level` argument
    - solution of stage 1 required before stage 2 can be built and solved
- The planner has additional problem instance parameters that are optional or required depending on stage
- Stage 1 may take (optional) accepted offers, stage 2 requires a flexibility request and may take open offers. The blocks that are built from these are defined in [economic_dispatch/model/flexibility.py](economic_dispatch/model/flexibility.py)
- The methods that are used to set or add problem instance parameters are:
    - `set_param`: stage 1, required unit specific parameters
    - `add_lfm_result`: stage 1, optional
    - `set_request`: stage 2, required
    - `add_open_offer`: stage 2, optional


## Simulation
- [economic_dispatch/simulation/component.py](economic_dispatch/simulation/component.py)
- Simulation component classes:
    - `EconomicDispatch`
        - `model` is an `EconomicDispatchPlanner` object
        - expects problem instance parameters before processing epoch
        - `process_epoch` solves the ED problem and sends results
    - `EconomicDispatchFlex`
        - `model` is an `EconomicDispatchFlexPlanner` object
        - expects problem instance parameters before processing stage 1 of epoch
        - `process_epoch` in stage 1 solves the ED problem and sends results
        - expects a flexibility request or a ready message which signals that no request is coming in stage 2 of epoch
        - `process_epoch` in stage 2 calculates a feasible set of offers to a flexibility request
- If participating market id is provided component creator builds `EconomicDispatchFlex`, otherwise `EconomicDispatch`

Environment variables specific to this component are listed below (see [economic_dispatch/component_manifest.yml](economic_dispatch/component_manifest.yml) for environment variable naming):
- `DISPATCH_HORIZON`: iso format duration, default=`PT36H`
- `DISPATCH_TIMESTEP`: iso format duration, default=`PT1H`
- `DISPATCH_SOLVER`: string specifying the solver for pyomo SolverFactory, default=`glpk`
- `MARKET_ID`: participating market id, default=`None`
- `RESOURCES`: resources and their types

Topics where results are sent are
- `DISPATCH_TOPIC`: topic for dispatch results, completed with `RESOURCE_TYPE`, default=`ResourceForecastState`
- `OFFER_TOPIC`: topic for offers to flexibility requests, completed with `MARKET_ID`, default=`Offer`

Listened topics are
- `RESOURCE_FORECAST_TOPIC`: topic for power forecasts, completed with `#`, default=`ResourceForecastState.Load,ResourceForecastState.Generator`
- `PRICE_FORECAST_STATE_TOPIC`: topic for price forecasts, completed with `#`, default=`PriceForecastState`
- `RESOURCE_STATE_TOPIC`: topic for storage states, completed with `#`, default=`ResourceState.Storage`
- `STATUS_TOPIC`: topic for ready messages (LFM ready signals), default=`Status.Ready`
- `LFM_RESULT_TOPIC`: topic for accepted offers, completed with `#`, default=`LFMMarketResult`
- `REQUEST_TOPIC`: topic for flexibility requests, completed with `#`, default=`Request`
- `CUSTOMER_INFO_TOPIC`: topic for resource id and cutomer id mappings, default=`Init.CIS.CustomerInfo`

Additional parameters

- `COMMITMENT_TIME`: Day-ahead market commitment time as a string, e.g. `12:00` which commits to next day market at twelve o'clock with next day being from midnight to midnight
- `WEIGHTS`: Storage component optimisation weights. See wiki for start message

The component can be launched with

    python -m economic_dispatch.simulation.component

Docker can be used with the included Dockerfile.

## Tests

The included unittests can be executed with:

    python -m unittest

This requires RabbitMQ connection information provided via environment variables as required by the AbstractSimulationComponent class. Tests can also be executed with docker compose:

    docker-compose -f docker-compose-test.yml up --build
    
Clean up after executing tests:

    docker-compose -f docker-compose-test.yml down -v


