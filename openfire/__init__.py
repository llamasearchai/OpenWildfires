"""
OpenFire: Advanced AI-Powered Drone Wildfire Detection Platform

A cutting-edge wildfire detection and monitoring platform that leverages
modern drone technology, advanced computer vision, and OpenAI's AI capabilities.

Author: Nik Jois <nikjois@llamasearch.ai>
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "MIT"

from openfire.detection import FireDetector, SmokeDetector
from openfire.drone import DroneController, FleetManager
from openfire.ai import OpenAIAnalyzer
from openfire.geolocation import LocationEstimator
from openfire.alerts import AlertSystem

__all__ = [
    "FireDetector",
    "SmokeDetector", 
    "DroneController",
    "FleetManager",
    "OpenAIAnalyzer",
    "LocationEstimator",
    "AlertSystem",
] 