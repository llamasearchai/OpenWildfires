"""
Advanced location estimation for fire detections.
"""

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class LocationEstimator:
    """Advanced location estimator for fire detections."""
    
    def __init__(self):
        self.earth_radius = 6371000  # Earth's radius in meters
        
    async def estimate_location(
        self,
        detection: 'Detection',
        drone_location: Tuple[float, float, float],  # (lat, lon, alt)
        camera_params: Dict[str, Any],
        terrain_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[float, float]]:
        """Estimate the geographic location of a fire detection."""
        try:
            # Extract detection center point
            center_x, center_y = detection.center
            
            # Get camera parameters
            fov = camera_params.get('fov', 60)  # Field of view in degrees
            resolution = camera_params.get('resolution', (1920, 1080))
            gimbal_pitch = camera_params.get('gimbal_pitch', -90)  # Degrees from horizontal
            gimbal_yaw = camera_params.get('gimbal_yaw', 0)  # Degrees from north
            
            # Calculate pixel angles
            pixel_angles = self._calculate_pixel_angles(
                center_x, center_y, resolution, fov
            )
            
            # Estimate ground intersection
            ground_location = self._estimate_ground_intersection(
                drone_location,
                pixel_angles,
                gimbal_pitch,
                gimbal_yaw,
                terrain_data
            )
            
            if ground_location:
                logger.info(
                    "Location estimated",
                    detection_center=detection.center,
                    estimated_location=ground_location,
                    drone_location=drone_location
                )
                
                return ground_location
            else:
                logger.warning("Failed to estimate ground intersection")
                return None
                
        except Exception as e:
            logger.error(f"Location estimation failed: {e}")
            return None
    
    def _calculate_pixel_angles(
        self,
        pixel_x: int,
        pixel_y: int,
        resolution: Tuple[int, int],
        fov: float
    ) -> Tuple[float, float]:
        """Calculate the angular offset of a pixel from the image center."""
        width, height = resolution
        
        # Convert pixel coordinates to normalized coordinates (-1 to 1)
        norm_x = (pixel_x - width / 2) / (width / 2)
        norm_y = (pixel_y - height / 2) / (height / 2)
        
        # Calculate angular offsets
        fov_rad = math.radians(fov)
        angle_x = norm_x * (fov_rad / 2)
        angle_y = norm_y * (fov_rad / 2)
        
        return angle_x, angle_y
    
    def _estimate_ground_intersection(
        self,
        drone_location: Tuple[float, float, float],
        pixel_angles: Tuple[float, float],
        gimbal_pitch: float,
        gimbal_yaw: float,
        terrain_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[float, float]]:
        """Estimate where the camera ray intersects the ground."""
        drone_lat, drone_lon, drone_alt = drone_location
        angle_x, angle_y = pixel_angles
        
        # Convert gimbal angles to radians
        pitch_rad = math.radians(gimbal_pitch)
        yaw_rad = math.radians(gimbal_yaw)
        
        # Calculate the direction vector of the camera ray
        # Assuming camera is pointing down (negative pitch)
        ray_direction = self._calculate_ray_direction(
            angle_x, angle_y, pitch_rad, yaw_rad
        )
        
        # Estimate ground elevation (use terrain data if available)
        ground_elevation = 0  # Default to sea level
        if terrain_data:
            ground_elevation = terrain_data.get('elevation', 0)
        
        # Calculate intersection with ground plane
        height_above_ground = drone_alt - ground_elevation
        
        if height_above_ground <= 0:
            logger.warning("Drone altitude below ground level")
            return None
        
        # Calculate horizontal distance to intersection
        if ray_direction[2] >= 0:  # Ray pointing up or horizontal
            logger.warning("Camera ray not pointing toward ground")
            return None
        
        # Distance along ray to ground intersection
        t = -height_above_ground / ray_direction[2]
        
        # Calculate horizontal displacement
        dx = ray_direction[0] * t  # East displacement in meters
        dy = ray_direction[1] * t  # North displacement in meters
        
        # Convert to lat/lon
        target_lat, target_lon = self._meters_to_latlon(
            drone_lat, drone_lon, dx, dy
        )
        
        return target_lat, target_lon
    
    def _calculate_ray_direction(
        self,
        angle_x: float,
        angle_y: float,
        pitch: float,
        yaw: float
    ) -> np.ndarray:
        """Calculate the 3D direction vector of the camera ray."""
        # Start with a ray pointing straight down
        ray = np.array([0, 0, -1])
        
        # Apply pixel offsets
        ray[0] = math.tan(angle_x)  # East component
        ray[1] = math.tan(angle_y)  # North component
        
        # Normalize
        ray = ray / np.linalg.norm(ray)
        
        # Apply gimbal rotations
        # Pitch rotation (around east axis)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        pitch_matrix = np.array([
            [1, 0, 0],
            [0, cos_pitch, -sin_pitch],
            [0, sin_pitch, cos_pitch]
        ])
        
        # Yaw rotation (around vertical axis)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        yaw_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Apply rotations
        ray = pitch_matrix @ ray
        ray = yaw_matrix @ ray
        
        return ray
    
    def _meters_to_latlon(
        self,
        origin_lat: float,
        origin_lon: float,
        dx: float,
        dy: float
    ) -> Tuple[float, float]:
        """Convert meter offsets to lat/lon coordinates."""
        # Convert to radians
        lat_rad = math.radians(origin_lat)
        
        # Calculate new coordinates
        new_lat = origin_lat + math.degrees(dy / self.earth_radius)
        new_lon = origin_lon + math.degrees(
            dx / (self.earth_radius * math.cos(lat_rad))
        )
        
        return new_lat, new_lon
    
    def calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two GPS coordinates using Haversine formula."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return self.earth_radius * c
    
    def calculate_bearing(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate bearing from point 1 to point 2."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        return (bearing_deg + 360) % 360
    
    async def estimate_fire_perimeter(
        self,
        detections: list,
        drone_locations: list,
        camera_params: Dict[str, Any]
    ) -> Optional[list]:
        """Estimate fire perimeter from multiple detections."""
        try:
            fire_points = []
            
            # Estimate location for each fire detection
            for detection, drone_location in zip(detections, drone_locations):
                if detection.class_name == "fire":
                    location = await self.estimate_location(
                        detection, drone_location, camera_params
                    )
                    if location:
                        fire_points.append(location)
            
            if len(fire_points) < 3:
                logger.warning("Insufficient fire points for perimeter estimation")
                return None
            
            # Calculate convex hull for fire perimeter
            perimeter = self._calculate_convex_hull(fire_points)
            
            logger.info(f"Fire perimeter estimated with {len(perimeter)} points")
            return perimeter
            
        except Exception as e:
            logger.error(f"Fire perimeter estimation failed: {e}")
            return None
    
    def _calculate_convex_hull(self, points: list) -> list:
        """Calculate convex hull of fire points."""
        try:
            from scipy.spatial import ConvexHull
            
            # Convert to numpy array
            points_array = np.array(points)
            
            # Calculate convex hull
            hull = ConvexHull(points_array)
            
            # Return hull vertices
            return [points[i] for i in hull.vertices]
            
        except ImportError:
            logger.warning("SciPy not available, using simple bounding box")
            return self._calculate_bounding_box(points)
        except Exception as e:
            logger.error(f"Convex hull calculation failed: {e}")
            return self._calculate_bounding_box(points)
    
    def _calculate_bounding_box(self, points: list) -> list:
        """Calculate simple bounding box as fallback."""
        if not points:
            return []
        
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Return bounding box corners
        return [
            (min_lat, min_lon),
            (min_lat, max_lon),
            (max_lat, max_lon),
            (max_lat, min_lon)
        ] 