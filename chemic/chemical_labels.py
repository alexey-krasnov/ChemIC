"""
Chemical Labels Module

This module defines a NamedTuple for chemical labels used in image classification.
The ChemicalLabels NamedTuple includes labels for various types of chemical structures
that can be identified in images. The order of the class labels in the NamedTuple is essential.

Classes:
    ChemicalLabels: A NamedTuple that holds labels for single chemical structure, chemical reactions,
                    no chemical structures, and multiple chemical structures.

Usage:
    Import the module and use the `chem_labels` instance to access the labels.

    Example:
        from chemical_labels import chem_labels
        print(chem_labels.single_chemical_structure)
        # Output: 'single chemical structure'

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: December 4, 2023
"""

from typing import NamedTuple


# Define class labels. Order of class label in the NamedTuple is essential!
class ChemicalLabels(NamedTuple):
    """"Class label for image classifier"""
    single_chemical_structure: str
    chemical_reactions: str
    no_chemical_structures: str
    multiple_chemical_structures: str


# Creating an instance of ChemicalLabels
chem_labels = ChemicalLabels(single_chemical_structure='single chemical structure',
                             chemical_reactions='chemical reactions',
                             no_chemical_structures='no chemical structures',
                             multiple_chemical_structures='multiple chemical structures')
