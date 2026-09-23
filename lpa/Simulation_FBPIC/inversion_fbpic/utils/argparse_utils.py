"""
argparse_utils.py

Utility functions for command-line argument parsing with argparse.

This module provides custom type functions and utilities for argparse that extend
the standard argparse functionality, such as handling None values from command-line
arguments.
"""

from typing import Optional


def float_or_none(value: str) -> Optional[float]:
    """
    Custom type for argparse that accepts float values or 'None'/'null' strings.

    This function allows command-line arguments to explicitly accept None values
    by passing the string 'None' or 'null' (case-insensitive). This is useful
    when you want to distinguish between "argument not provided" (use default)
    and "argument explicitly set to None" (override default).

    Args:
        value: String value from command line

    Returns:
        float value or None if 'None' or 'null' is provided

    Examples:
        >>> float_or_none("3.14")
        3.14
        >>> float_or_none("None")
        None
        >>> float_or_none("null")
        None
        >>> float_or_none("NONE")
        None
    """
    if value.lower() in ("none", "null"):
        return None
    return float(value)


def build_selection_dict(
    min_uz: Optional[float] = None,
    max_uz: Optional[float] = None,
    min_z: Optional[float] = None,
    max_z: Optional[float] = None,
) -> Optional[dict]:
    """
    Build selection dictionary from individual parameters for particle selection.

    This function constructs a selection dictionary suitable for use with particle
    loading functions (e.g., LpaDiagnostics.get_particle). The dictionary format
    is: {"uz": [min_uz, max_uz], "z": [min_z, max_z]}.

    Args:
        min_uz: Minimum uz value.
        max_uz: Maximum uz value.
        min_z: Minimum z value.
        max_z: Maximum z value.

    Returns:
        Selection dictionary with "uz" and/or "z" keys, or None if all values are None.

    Examples:
        >>> build_selection_dict(min_uz=200, min_z=1e-6)
        {'uz': [200, None], 'z': [1e-06, None]}

        >>> build_selection_dict(min_uz=200, max_uz=500)
        {'uz': [200, 500]}

        >>> build_selection_dict()
        None
    """
    # Build selection dict
    selection = {}
    if min_uz is not None or max_uz is not None:
        selection["uz"] = [min_uz, max_uz]
    if min_z is not None or max_z is not None:
        selection["z"] = [min_z, max_z]

    return selection if selection else None
