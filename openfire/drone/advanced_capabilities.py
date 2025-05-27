"""
Advanced Drone Capabilities for Wildfire Detection

This module implements cutting-edge drone-only technologies for wildfire detection
and monitoring, showcasing autonomous swarm intelligence, multi-spectral analysis,
and real-time coordination without any satellite dependencies.

Author: Nik Jois <nikjois@llamasearch.ai>
License: MIT
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from ..ai.analyzer import OpenAIAnalyzer
from ..detection.models import FireDetector
from .controller import DroneController


@dataclass
class DroneSwarmNode:
    """Individual drone node in the swarm network."""
    drone_id: str
    controller: DroneController
    position: Tuple[float, float, float]  # lat, lon, alt
    battery_level: float
    sensor_capabilities: List[str]
    communication_range: float
    last_update: datetime


@dataclass
class FireHotspot:
    """Detected fire hotspot with drone-derived intelligence."""
    location: Tuple[float, float]
    intensity: float
    size_estimate: float
    growth_rate: float
    threat_level: str
    detection_confidence: float
    first_detected: datetime
    last_confirmed: datetime
    detecting_drones: List[str]


class DroneSwarmIntelligence:
    """
    Advanced swarm intelligence system for coordinated wildfire detection.
    
    This system implements autonomous drone coordination, distributed sensing,
    and collective intelligence for comprehensive wildfire monitoring using
    only drone-based technologies.
    """
    
    def __init__(self, ai_analyzer: OpenAIAnalyzer, fire_detector: FireDetector):
        self.ai_analyzer = ai_analyzer
        self.fire_detector = fire_detector
        self.swarm_nodes: Dict[str, DroneSwarmNode] = {}
        self.active_hotspots: Dict[str, FireHotspot] = {}
        self.coverage_grid: np.ndarray = None
        self.coordination_network = {}
        
    async def register_drone(self, drone_controller: DroneController, 
                           capabilities: List[str]) -> str:
        """Register a new drone in the swarm network."""
        drone_id = drone_controller.drone_id
        
        # Get initial position and status
        position = await drone_controller.get_gps_location()
        battery = await drone_controller.get_battery_level()
        
        node = DroneSwarmNode(
            drone_id=drone_id,
            controller=drone_controller,
            position=position,
            battery_level=battery,
            sensor_capabilities=capabilities,
            communication_range=5000.0,  # 5km range
            last_update=datetime.utcnow()
        )
        
        self.swarm_nodes[drone_id] = node
        await self._update_coordination_network()
        
        return drone_id
    
    async def autonomous_area_coverage(self, area_bounds: Dict[str, float],
                                     coverage_resolution: float = 100.0) -> Dict[str, Any]:
        """
        Implement autonomous area coverage using swarm intelligence.
        
        Args:
            area_bounds: Dictionary with 'north', 'south', 'east', 'west' bounds
            coverage_resolution: Grid resolution in meters
            
        Returns:
            Coverage plan and execution status
        """
        # Calculate optimal drone distribution
        coverage_plan = await self._calculate_optimal_coverage(area_bounds, coverage_resolution)
        
        # Assign waypoints to each drone
        drone_assignments = await self._assign_coverage_waypoints(coverage_plan)
        
        # Execute coordinated coverage mission
        execution_tasks = []
        for drone_id, waypoints in drone_assignments.items():
            if drone_id in self.swarm_nodes:
                task = self._execute_coverage_mission(drone_id, waypoints)
                execution_tasks.append(task)
        
        # Monitor execution
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        return {
            "coverage_plan": coverage_plan,
            "drone_assignments": drone_assignments,
            "execution_results": results,
            "coverage_percentage": await self._calculate_coverage_percentage(area_bounds)
        }
    
    async def multi_spectral_fire_analysis(self, drone_id: str, 
                                         image_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform advanced multi-spectral fire analysis using drone sensors.
        
        Args:
            drone_id: ID of the detecting drone
            image_data: Dictionary containing RGB, thermal, and NIR images
            
        Returns:
            Comprehensive fire analysis results
        """
        analysis_results = {}
        
        # RGB analysis for smoke and visible flames
        if 'rgb' in image_data:
            rgb_analysis = await self.fire_detector.detect(image_data['rgb'])
            analysis_results['rgb_detection'] = rgb_analysis
        
        # Thermal analysis for heat signatures
        if 'thermal' in image_data:
            thermal_analysis = await self._analyze_thermal_signature(image_data['thermal'])
            analysis_results['thermal_analysis'] = thermal_analysis
        
        # Near-infrared analysis for vegetation stress
        if 'nir' in image_data:
            nir_analysis = await self._analyze_vegetation_stress(image_data['nir'])
            analysis_results['vegetation_stress'] = nir_analysis
        
        # Combine multi-spectral data for enhanced detection
        combined_analysis = await self._fuse_spectral_data(analysis_results)
        
        # OpenAI enhanced scene understanding
        scene_context = await self.ai_analyzer.analyze_scene(
            image=image_data.get('rgb'),
            context="Multi-spectral wildfire analysis from drone platform",
            include_risk_assessment=True
        )
        
        return {
            "drone_id": drone_id,
            "spectral_analysis": analysis_results,
            "combined_detection": combined_analysis,
            "ai_scene_analysis": scene_context,
            "confidence_score": combined_analysis.get('confidence', 0.0),
            "threat_assessment": self._calculate_threat_level(combined_analysis, scene_context)
        }
    
    async def adaptive_flight_planning(self, environmental_conditions: Dict[str, Any],
                                     fire_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Implement adaptive flight planning based on real-time conditions.
        
        Args:
            environmental_conditions: Weather and terrain data
            fire_locations: Known fire locations for priority coverage
            
        Returns:
            Optimized flight plans for each drone
        """
        flight_plans = {}
        
        for drone_id, node in self.swarm_nodes.items():
            # Calculate wind-optimized flight paths
            wind_vector = (
                environmental_conditions.get('wind_speed', 0),
                environmental_conditions.get('wind_direction', 0)
            )
            
            # Optimize for battery efficiency and coverage
            optimized_path = await self._optimize_flight_path(
                drone_id, wind_vector, fire_locations, node.battery_level
            )
            
            # Add dynamic waypoint adjustment
            adaptive_waypoints = await self._add_adaptive_waypoints(
                optimized_path, environmental_conditions
            )
            
            flight_plans[drone_id] = {
                "base_path": optimized_path,
                "adaptive_waypoints": adaptive_waypoints,
                "estimated_flight_time": self._estimate_flight_time(adaptive_waypoints),
                "battery_consumption": self._estimate_battery_usage(adaptive_waypoints),
                "coverage_priority": self._calculate_coverage_priority(fire_locations)
            }
        
        return flight_plans
    
    async def real_time_swarm_coordination(self) -> Dict[str, Any]:
        """
        Implement real-time swarm coordination and communication.
        
        Returns:
            Coordination status and network health
        """
        coordination_status = {
            "network_health": await self._assess_network_health(),
            "active_drones": len(self.swarm_nodes),
            "communication_matrix": await self._build_communication_matrix(),
            "task_distribution": await self._optimize_task_distribution(),
            "emergency_protocols": await self._check_emergency_protocols()
        }
        
        # Update drone positions and status
        for drone_id, node in self.swarm_nodes.items():
            try:
                position = await node.controller.get_gps_location()
                battery = await node.controller.get_battery_level()
                
                node.position = position
                node.battery_level = battery
                node.last_update = datetime.utcnow()
                
            except Exception as e:
                coordination_status["errors"] = coordination_status.get("errors", [])
                coordination_status["errors"].append(f"Drone {drone_id}: {str(e)}")
        
        # Implement dynamic task reallocation
        if coordination_status["network_health"] < 0.8:
            await self._implement_emergency_reallocation()
        
        return coordination_status
    
    async def edge_computing_fire_detection(self, drone_id: str, 
                                          sensor_stream: Any) -> Dict[str, Any]:
        """
        Implement edge computing for real-time fire detection on drones.
        
        Args:
            drone_id: ID of the processing drone
            sensor_stream: Real-time sensor data stream
            
        Returns:
            Real-time detection results
        """
        edge_results = {
            "processing_node": drone_id,
            "processing_time": datetime.utcnow(),
            "detections": [],
            "system_performance": {}
        }
        
        # Process sensor stream in real-time
        async for frame_data in sensor_stream:
            start_time = datetime.utcnow()
            
            # Lightweight detection for edge processing
            detection = await self.fire_detector.detect_lightweight(frame_data)
            
            if detection.confidence > 0.7:  # High confidence threshold
                # Immediate alert for critical detections
                await self._trigger_immediate_alert(drone_id, detection)
                
                # Add to results
                edge_results["detections"].append({
                    "timestamp": start_time,
                    "confidence": detection.confidence,
                    "location": await self.swarm_nodes[drone_id].controller.get_gps_location(),
                    "detection_type": detection.detection_type
                })
            
            # Track performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            edge_results["system_performance"]["avg_processing_time"] = processing_time
        
        return edge_results
    
    # Private helper methods
    
    async def _calculate_optimal_coverage(self, area_bounds: Dict[str, float], 
                                        resolution: float) -> np.ndarray:
        """Calculate optimal coverage grid for the area."""
        # Create coverage grid
        lat_range = area_bounds['north'] - area_bounds['south']
        lon_range = area_bounds['east'] - area_bounds['west']
        
        grid_size_lat = int(lat_range * 111000 / resolution)  # Convert to meters
        grid_size_lon = int(lon_range * 111000 / resolution)
        
        coverage_grid = np.zeros((grid_size_lat, grid_size_lon))
        
        # Apply priority weighting based on terrain and risk factors
        for i in range(grid_size_lat):
            for j in range(grid_size_lon):
                # Higher priority for forest areas, slopes, etc.
                coverage_grid[i, j] = await self._calculate_cell_priority(i, j, area_bounds)
        
        return coverage_grid
    
    async def _assign_coverage_waypoints(self, coverage_grid: np.ndarray) -> Dict[str, List]:
        """Assign optimal waypoints to each drone."""
        assignments = {}
        
        # Use clustering to assign grid cells to drones
        drone_positions = [node.position[:2] for node in self.swarm_nodes.values()]
        drone_ids = list(self.swarm_nodes.keys())
        
        if len(drone_positions) > 0:
            # Create waypoint clusters
            high_priority_cells = np.where(coverage_grid > 0.7)
            waypoints = list(zip(high_priority_cells[0], high_priority_cells[1]))
            
            if len(waypoints) > 0:
                # Assign waypoints to nearest drones
                distances = cdist(waypoints, drone_positions)
                assignments_idx = np.argmin(distances, axis=1)
                
                for i, drone_idx in enumerate(assignments_idx):
                    drone_id = drone_ids[drone_idx]
                    if drone_id not in assignments:
                        assignments[drone_id] = []
                    assignments[drone_id].append(waypoints[i])
        
        return assignments
    
    async def _analyze_thermal_signature(self, thermal_image: np.ndarray) -> Dict[str, Any]:
        """Analyze thermal signature for fire detection."""
        # Temperature threshold analysis
        hot_pixels = thermal_image > 50  # Celsius threshold
        
        # Connected component analysis
        num_labels, labels = cv2.connectedComponents(hot_pixels.astype(np.uint8))
        
        hotspots = []
        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) > 10:  # Minimum size threshold
                hotspots.append({
                    "area": np.sum(mask),
                    "max_temp": np.max(thermal_image[mask]),
                    "avg_temp": np.mean(thermal_image[mask]),
                    "centroid": np.mean(np.where(mask), axis=1)
                })
        
        return {
            "hotspot_count": len(hotspots),
            "hotspots": hotspots,
            "max_temperature": np.max(thermal_image),
            "thermal_anomalies": len(hotspots) > 0
        }
    
    async def _analyze_vegetation_stress(self, nir_image: np.ndarray) -> Dict[str, Any]:
        """Analyze vegetation stress using near-infrared data."""
        # Calculate NDVI-like index for vegetation health
        # Simulated calculation for demonstration
        vegetation_index = np.mean(nir_image) / 255.0
        
        stress_areas = nir_image < (np.mean(nir_image) - 2 * np.std(nir_image))
        
        return {
            "vegetation_health_index": vegetation_index,
            "stress_area_percentage": np.sum(stress_areas) / stress_areas.size,
            "drought_indicators": vegetation_index < 0.3,
            "fire_susceptibility": "high" if vegetation_index < 0.2 else "medium" if vegetation_index < 0.4 else "low"
        }
    
    async def _fuse_spectral_data(self, spectral_results: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse multi-spectral analysis results."""
        confidence_scores = []
        detection_types = []
        
        # Combine confidence scores from different spectral bands
        if 'rgb_detection' in spectral_results:
            confidence_scores.append(spectral_results['rgb_detection'].confidence)
            detection_types.append(spectral_results['rgb_detection'].detection_type)
        
        if 'thermal_analysis' in spectral_results:
            thermal_confidence = 0.9 if spectral_results['thermal_analysis']['thermal_anomalies'] else 0.1
            confidence_scores.append(thermal_confidence)
        
        if 'vegetation_stress' in spectral_results:
            stress_confidence = 0.8 if spectral_results['vegetation_stress']['drought_indicators'] else 0.2
            confidence_scores.append(stress_confidence)
        
        # Calculate combined confidence using weighted average
        combined_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            "confidence": combined_confidence,
            "detection_type": "fire" if combined_confidence > 0.7 else "smoke" if combined_confidence > 0.4 else "none",
            "spectral_agreement": len(confidence_scores),
            "multi_band_detection": combined_confidence > 0.6
        }
    
    def _calculate_threat_level(self, detection_results: Dict[str, Any], 
                              scene_analysis: Any) -> str:
        """Calculate overall threat level."""
        confidence = detection_results.get('confidence', 0.0)
        
        if confidence > 0.8:
            return "critical"
        elif confidence > 0.6:
            return "high"
        elif confidence > 0.4:
            return "medium"
        else:
            return "low" 