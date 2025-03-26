"""
Configuration package for Finite Monkey Engine.
DEPRECATED: Use ../nodes_config.py instead.
"""

import logging
import warnings

logger = logging.getLogger(__name__)
warnings.warn(
    "The finite_monkey.config package is deprecated. Use finite_monkey.nodes_config instead.",
    DeprecationWarning,
    stacklevel=2
)
