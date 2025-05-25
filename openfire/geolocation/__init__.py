"""
Advanced geolocation and mapping module.

This module provides precise location estimation for fire detections
using drone telemetry, camera parameters, and terrain data.
"""

from openfire.geolocation.estimator import LocationEstimator
from openfire.geolocation.mapping import MapProjector

__all__ = [
    "LocationEstimator",
    "MapProjector",
] 