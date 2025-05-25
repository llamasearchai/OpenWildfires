"""
Advanced drone controller with MAVLink integration.
"""

import asyncio
import time
from typing import Optional, Tuple, Dict, Any, AsyncGenerator
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import structlog

from openfire.config import get_settings

logger = structlog.get_logger(__name__)


class DroneStatus:
    """Represents the current status of a drone."""
    
    def __init__(self):
        self.is_connected = False
        self.is_armed = False
        self.mode = "UNKNOWN"
        self.battery_level = 0.0
        self.gps_location = None
        self.altitude = 0.0
        self.heading = 0.0
        self.velocity = (0.0, 0.0, 0.0)  # (vx, vy, vz)
        self.last_heartbeat = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "is_connected": self.is_connected,
            "is_armed": self.is_armed,
            "mode": self.mode,
            "battery_level": self.battery_level,
            "gps_location": {
                "lat": self.gps_location.lat if self.gps_location else None,
                "lon": self.gps_location.lon if self.gps_location else None,
                "alt": self.gps_location.alt if self.gps_location else None
            } if self.gps_location else None,
            "altitude": self.altitude,
            "heading": self.heading,
            "velocity": self.velocity,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None
        }


class DroneController:
    """Advanced drone controller with autonomous capabilities."""
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        drone_id: str = "drone_001"
    ):
        self.settings = get_settings()
        self.connection_string = connection_string or self.settings.drone.connection_string
        self.drone_id = drone_id
        self.vehicle = None
        self.status = DroneStatus()
        self._camera_stream = None
        self._mission_active = False
        self._emergency_stop = False
        
    async def connect(self) -> bool:
        """Connect to the drone."""
        try:
            logger.info(f"Connecting to drone at {self.connection_string}")
            
            # Connect to vehicle
            self.vehicle = connect(
                self.connection_string,
                wait_ready=True,
                timeout=self.settings.drone.timeout
            )
            
            # Set up status monitoring
            await self._setup_status_monitoring()
            
            self.status.is_connected = True
            logger.info("Successfully connected to drone", drone_id=self.drone_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to drone: {e}", drone_id=self.drone_id)
            self.status.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the drone."""
        try:
            if self.vehicle:
                self.vehicle.close()
                self.vehicle = None
            
            self.status.is_connected = False
            logger.info("Disconnected from drone", drone_id=self.drone_id)
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}", drone_id=self.drone_id)
    
    async def arm(self) -> bool:
        """Arm the drone."""
        if not self.status.is_connected:
            logger.error("Cannot arm: drone not connected")
            return False
        
        try:
            # Pre-arm checks
            if not await self._pre_arm_checks():
                return False
            
            # Arm the vehicle
            self.vehicle.armed = True
            
            # Wait for arming
            timeout = 10
            start_time = time.time()
            while not self.vehicle.armed and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if self.vehicle.armed:
                self.status.is_armed = True
                logger.info("Drone armed successfully", drone_id=self.drone_id)
                return True
            else:
                logger.error("Failed to arm drone within timeout")
                return False
                
        except Exception as e:
            logger.error(f"Failed to arm drone: {e}", drone_id=self.drone_id)
            return False
    
    async def disarm(self) -> bool:
        """Disarm the drone."""
        try:
            if self.vehicle and self.vehicle.armed:
                self.vehicle.armed = False
                
                # Wait for disarming
                timeout = 5
                start_time = time.time()
                while self.vehicle.armed and (time.time() - start_time) < timeout:
                    await asyncio.sleep(0.1)
                
                self.status.is_armed = False
                logger.info("Drone disarmed", drone_id=self.drone_id)
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to disarm drone: {e}", drone_id=self.drone_id)
            return False
    
    async def takeoff(self, altitude: float = 10.0) -> bool:
        """Take off to specified altitude."""
        if not self.status.is_armed:
            logger.error("Cannot takeoff: drone not armed")
            return False
        
        try:
            # Set mode to GUIDED
            self.vehicle.mode = VehicleMode("GUIDED")
            await asyncio.sleep(1)
            
            # Takeoff
            logger.info(f"Taking off to {altitude}m", drone_id=self.drone_id)
            self.vehicle.simple_takeoff(altitude)
            
            # Wait for takeoff to complete
            while True:
                current_altitude = self.vehicle.location.global_relative_frame.alt
                if current_altitude >= altitude * 0.95:
                    break
                await asyncio.sleep(1)
            
            logger.info("Takeoff complete", drone_id=self.drone_id, altitude=current_altitude)
            return True
            
        except Exception as e:
            logger.error(f"Takeoff failed: {e}", drone_id=self.drone_id)
            return False
    
    async def land(self) -> bool:
        """Land the drone."""
        try:
            logger.info("Landing drone", drone_id=self.drone_id)
            
            # Set mode to LAND
            self.vehicle.mode = VehicleMode("LAND")
            
            # Wait for landing
            while self.vehicle.armed:
                await asyncio.sleep(1)
            
            logger.info("Landing complete", drone_id=self.drone_id)
            return True
            
        except Exception as e:
            logger.error(f"Landing failed: {e}", drone_id=self.drone_id)
            return False
    
    async def goto_location(
        self,
        lat: float,
        lon: float,
        alt: float,
        groundspeed: float = 5.0
    ) -> bool:
        """Fly to a specific GPS location."""
        try:
            # Set groundspeed
            self.vehicle.groundspeed = groundspeed
            
            # Create target location
            target_location = LocationGlobalRelative(lat, lon, alt)
            
            logger.info(
                f"Flying to location",
                drone_id=self.drone_id,
                target_lat=lat,
                target_lon=lon,
                target_alt=alt
            )
            
            # Go to location
            self.vehicle.simple_goto(target_location)
            
            # Wait for arrival (simplified - in practice you'd want more sophisticated logic)
            while True:
                current_location = self.vehicle.location.global_relative_frame
                distance = self._calculate_distance(
                    current_location.lat, current_location.lon,
                    lat, lon
                )
                
                if distance < 2.0:  # Within 2 meters
                    break
                    
                await asyncio.sleep(1)
            
            logger.info("Arrived at target location", drone_id=self.drone_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to reach location: {e}", drone_id=self.drone_id)
            return False
    
    async def emergency_stop(self) -> None:
        """Emergency stop - immediately land the drone."""
        logger.warning("EMERGENCY STOP activated", drone_id=self.drone_id)
        self._emergency_stop = True
        self._mission_active = False
        
        try:
            # Set to LAND mode immediately
            if self.vehicle:
                self.vehicle.mode = VehicleMode("LAND")
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}", drone_id=self.drone_id)
    
    async def get_gps_location(self) -> Optional[Tuple[float, float, float]]:
        """Get current GPS location."""
        if not self.vehicle:
            return None
        
        try:
            location = self.vehicle.location.global_relative_frame
            return (location.lat, location.lon, location.alt)
        except Exception as e:
            logger.error(f"Failed to get GPS location: {e}")
            return None
    
    async def camera_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """Stream camera frames from the drone."""
        # This is a mock implementation - in practice you'd connect to actual camera
        cap = cv2.VideoCapture(0)  # Use webcam for demo
        
        try:
            while self.status.is_connected and not self._emergency_stop:
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield frame_rgb
                
                await asyncio.sleep(0.033)  # ~30 FPS
                
        finally:
            cap.release()
    
    async def start_patrol_mission(
        self,
        waypoints: list,
        altitude: float = 50.0
    ) -> None:
        """Start an autonomous patrol mission."""
        if not self.status.is_armed:
            logger.error("Cannot start mission: drone not armed")
            return
        
        self._mission_active = True
        logger.info(
            f"Starting patrol mission with {len(waypoints)} waypoints",
            drone_id=self.drone_id
        )
        
        try:
            for i, (lat, lon) in enumerate(waypoints):
                if not self._mission_active or self._emergency_stop:
                    break
                
                logger.info(f"Flying to waypoint {i+1}/{len(waypoints)}")
                await self.goto_location(lat, lon, altitude)
                
                # Hover for observation
                await asyncio.sleep(5)
            
            logger.info("Patrol mission completed", drone_id=self.drone_id)
            
        except Exception as e:
            logger.error(f"Mission failed: {e}", drone_id=self.drone_id)
        finally:
            self._mission_active = False
    
    async def stop_mission(self) -> None:
        """Stop the current mission."""
        self._mission_active = False
        logger.info("Mission stopped", drone_id=self.drone_id)
    
    def get_status(self) -> DroneStatus:
        """Get current drone status."""
        if self.vehicle:
            self.status.mode = str(self.vehicle.mode.name)
            self.status.is_armed = self.vehicle.armed
            
            if self.vehicle.battery:
                self.status.battery_level = self.vehicle.battery.level
            
            if self.vehicle.location.global_relative_frame:
                self.status.gps_location = self.vehicle.location.global_relative_frame
                self.status.altitude = self.vehicle.location.global_relative_frame.alt
            
            if self.vehicle.heading:
                self.status.heading = self.vehicle.heading
            
            if self.vehicle.velocity:
                self.status.velocity = (
                    self.vehicle.velocity[0],
                    self.vehicle.velocity[1], 
                    self.vehicle.velocity[2]
                )
            
            self.status.last_heartbeat = time.time()
        
        return self.status
    
    async def _setup_status_monitoring(self) -> None:
        """Set up continuous status monitoring."""
        # This would set up callbacks for vehicle parameter changes
        # For now, we'll rely on polling in get_status()
        pass
    
    async def _pre_arm_checks(self) -> bool:
        """Perform pre-arm safety checks."""
        try:
            # Check GPS fix
            if not self.vehicle.gps_0.fix_type or self.vehicle.gps_0.fix_type < 3:
                logger.error("GPS fix not available")
                return False
            
            # Check battery level
            if self.vehicle.battery and self.vehicle.battery.level < self.settings.drone.min_battery:
                logger.error(f"Battery too low: {self.vehicle.battery.level}%")
                return False
            
            # Check if vehicle is ready
            if not self.vehicle.is_armable:
                logger.error("Vehicle not ready for arming")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pre-arm check failed: {e}")
            return False
    
    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two GPS coordinates in meters."""
        # Haversine formula
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c


def main():
    """Main function for standalone drone controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenFire Drone Controller")
    parser.add_argument("--connection", help="MAVLink connection string")
    parser.add_argument("--drone-id", default="drone_001", help="Drone ID")
    
    args = parser.parse_args()
    
    async def run_controller():
        controller = DroneController(args.connection, args.drone_id)
        
        if await controller.connect():
            logger.info("Drone controller ready")
            
            # Keep running until interrupted
            try:
                while True:
                    status = controller.get_status()
                    logger.info("Drone status", **status.to_dict())
                    await asyncio.sleep(5)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
            finally:
                await controller.disconnect()
    
    asyncio.run(run_controller())


if __name__ == "__main__":
    main() 