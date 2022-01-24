# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains base class for unit level components of the economic dispatch.
"""


class _Unit:
    """
    Base unit class

    Attributes
    ----------
    name: str
        name of the unit

    Methods
    -------
    block_rule(block: pyo.Block)
        Builds all the optimisation model components for this unit type 
        on top of the passed block.
    topics(prefix: bool=True)
        returns a list of unit's problem instance parameter names, 
        if prefix=True '{self.name}.' is added before attribute names.
    """

    PROB_INSTANCE_PARAMS = []
    TIME_INDEXED_PROB_INSTANCE_PARAMS = []

    def __init__(self, name):
        """
        Parameters
        ----------
        name: str
            name of the unit
        """
        self.name = name

    @property
    def _required_inputs(self):
        """ List of problem instance parameter names for this unit. """
        return self.PROB_INSTANCE_PARAMS

    def non_time_indexed_parameters(self):
        """ List of names of parameter that are not time indexed in the model for this unit type. """
        return [p for p in self.PROB_INSTANCE_PARAMS if p not in self.TIME_INDEXED_PROB_INSTANCE_PARAMS]

    def time_indexed_parameters(self):
        """ List of names of parameter that are time indexed in the model for this unit type. """
        return self.TIME_INDEXED_PROB_INSTANCE_PARAMS

    def clear(self):
        """ Clears problem instance parameter values. """
        for param in self.non_time_indexed_parameters():
            setattr(self, param, None)
        for param in self.time_indexed_parameters():
            setattr(self, param, [])

    def ready(self):
        """ Returns True if all problem instance parameter values are ready. 
        
        Not ready if no time indexed param is None or
        time indexed param is empty
        """
        return not any(getattr(self, param) is None for param in self.non_time_indexed_parameters()) \
            and not any(len(getattr(self, param)) == 0 for param in self.time_indexed_parameters())

    def topics(self, prefix=True):
        """ Return a list of unit's problem instance parameter names.

        If prefix=True '{self.name}.' is added before attribute names.
        """
        if prefix:
            return [self.name + '.' + t for t in self._required_inputs]
        else:
            return [t for t in self._required_inputs]

    def __repr__(self):
        return self.__class__.__name__ + '(name=\'' + self.name + '\')'

    def block_rule(self, block):
        """
        Builds all the optimisation model units for this unit type
        on top of the passed block.

        Parameters
        ----------
        block: pyo.Block
            reference to a unit level block of the model.
        """

        pass
