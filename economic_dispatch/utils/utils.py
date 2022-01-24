# -*- coding: utf-8 -*-

# Copyright 2021 Tampere University and VTT Technical Research Centre of Finland
# This software was developed as a part of the ProCemPlus project: https://www.senecc.fi/projects/procemplus
# This source code is licensed under the MIT license. See LICENSE in the repository root directory.
# Author(s): Olli Suominen <olli.suominen@tuni.fi>
#            Ville Mörsky <ville.morsky@tuni.fi>

"""
Contains utility classes and functions.
"""


class Barrier:
    """
    Barrier object to check that all needed parameter for the model have been set.

    Attributes
    ----------
    full_list: List[str]
        list of everything that is needed
    missing: List[str]
        list of what is still missing
    pass_state: bool
        False if something missing

    Methods
    -------
    process_arrive(id)
        remove id from missing
    reset()
        sets missing to everything in full_list
    """

    def __init__(self, ids):

        self.full_list = ids
        self.missing = ids[:]

    @property
    def pass_state(self):
        if self.missing:
            return False
        else:
            return True

    def process_arrive(self, id):
        if id in self.missing:
            self.missing.remove(id)
            return True
        else:
            return False

    def reset(self):
        self.missing = self.full_list[:]
