"""
SQLAlchemy database models for OpenWildfires platform.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    drones = relationship("Drone", back_populates="owner")
    missions = relationship("Mission", back_populates="created_by")
    alerts = relationship("Alert", back_populates="created_by")


class Drone(Base):
    """Drone model for tracking drone information and status."""
    
    __tablename__ = "drones"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drone_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    model = Column(String(100))
    serial_number = Column(String(100), unique=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_connected = Column(Boolean, default=False)
    is_armed = Column(Boolean, default=False)
    current_mode = Column(String(50))
    battery_level = Column(Float)
    
    # Location
    latitude = Column(Float)
    longitude = Column(Float)
    altitude = Column(Float)
    heading = Column(Float)
    
    # Capabilities
    max_altitude = Column(Float, default=120.0)
    max_speed = Column(Float, default=15.0)
    camera_resolution = Column(String(50))
    has_thermal_camera = Column(Boolean, default=False)
    has_lidar = Column(Boolean, default=False)
    
    # Ownership
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_seen = Column(DateTime)
    
    # Relationships
    owner = relationship("User", back_populates="drones")
    missions = relationship("Mission", back_populates="drone")
    detections = relationship("Detection", back_populates="drone")
    telemetry = relationship("TelemetryData", back_populates="drone")
    
    # Indexes
    __table_args__ = (
        Index("idx_drone_location", "latitude", "longitude"),
        Index("idx_drone_status", "is_active", "is_connected"),
    )


class Mission(Base):
    """Mission model for tracking drone missions."""
    
    __tablename__ = "missions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    mission_type = Column(String(50), nullable=False)  # patrol, search, emergency
    
    # Status
    status = Column(String(50), default="planned")  # planned, active, completed, aborted
    priority = Column(String(20), default="medium")  # low, medium, high, critical
    
    # Mission parameters
    waypoints = Column(JSON)  # List of [lat, lon, alt] coordinates
    flight_altitude = Column(Float, default=50.0)
    flight_speed = Column(Float, default=5.0)
    area_coverage = Column(JSON)  # Polygon coordinates for coverage area
    
    # Timing
    scheduled_start = Column(DateTime)
    actual_start = Column(DateTime)
    scheduled_end = Column(DateTime)
    actual_end = Column(DateTime)
    estimated_duration = Column(Integer)  # minutes
    
    # Assignment
    drone_id = Column(UUID(as_uuid=True), ForeignKey("drones.id"))
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    drone = relationship("Drone", back_populates="missions")
    created_by = relationship("User", back_populates="missions")
    detections = relationship("Detection", back_populates="mission")
    
    # Indexes
    __table_args__ = (
        Index("idx_mission_status", "status", "priority"),
        Index("idx_mission_timing", "scheduled_start", "actual_start"),
    )


class Detection(Base):
    """Detection model for storing fire and smoke detection results."""
    
    __tablename__ = "detections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Detection details
    detection_type = Column(String(20), nullable=False)  # fire, smoke
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON)  # [x1, y1, x2, y2]
    mask_data = Column(JSON)  # Segmentation mask if available
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    altitude = Column(Float)
    
    # Image data
    image_path = Column(String(500))
    image_metadata = Column(JSON)
    thermal_image_path = Column(String(500))
    
    # AI Analysis
    ai_analysis = Column(JSON)  # OpenAI analysis results
    risk_level = Column(String(20))  # low, medium, high, critical
    spread_prediction = Column(JSON)
    
    # Weather context
    weather_conditions = Column(JSON)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    
    # Verification
    is_verified = Column(Boolean, default=False)
    verified_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    verification_notes = Column(Text)
    
    # Assignment
    drone_id = Column(UUID(as_uuid=True), ForeignKey("drones.id"))
    mission_id = Column(UUID(as_uuid=True), ForeignKey("missions.id"))
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    drone = relationship("Drone", back_populates="detections")
    mission = relationship("Mission", back_populates="detections")
    verified_by = relationship("User")
    alerts = relationship("Alert", back_populates="detection")
    
    # Indexes
    __table_args__ = (
        Index("idx_detection_location", "latitude", "longitude"),
        Index("idx_detection_type_confidence", "detection_type", "confidence"),
        Index("idx_detection_time", "detected_at"),
        Index("idx_detection_risk", "risk_level", "confidence"),
    )


class Alert(Base):
    """Alert model for tracking emergency alerts and notifications."""
    
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # fire_detected, smoke_detected, emergency
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    priority = Column(String(20), default="medium")  # low, medium, high, critical
    
    # Status
    status = Column(String(20), default="active")  # active, acknowledged, resolved, dismissed
    acknowledged_at = Column(DateTime)
    acknowledged_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    resolved_at = Column(DateTime)
    
    # Location
    latitude = Column(Float)
    longitude = Column(Float)
    location_description = Column(String(500))
    
    # Notification tracking
    sms_sent = Column(Boolean, default=False)
    email_sent = Column(Boolean, default=False)
    push_sent = Column(Boolean, default=False)
    emergency_services_notified = Column(Boolean, default=False)
    
    # Metadata
    metadata = Column(JSON)
    escalation_level = Column(Integer, default=1)
    
    # Relationships
    detection_id = Column(UUID(as_uuid=True), ForeignKey("detections.id"))
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    detection = relationship("Detection", back_populates="alerts")
    created_by = relationship("User", back_populates="alerts")
    acknowledged_by = relationship("User", foreign_keys=[acknowledged_by_id])
    
    # Indexes
    __table_args__ = (
        Index("idx_alert_status_priority", "status", "priority"),
        Index("idx_alert_type_time", "alert_type", "created_at"),
        Index("idx_alert_location", "latitude", "longitude"),
    )


class Incident(Base):
    """Incident model for tracking wildfire incidents."""
    
    __tablename__ = "incidents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Incident details
    incident_number = Column(String(50), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    incident_type = Column(String(50), default="wildfire")
    
    # Status
    status = Column(String(50), default="active")  # active, contained, controlled, extinguished
    severity = Column(String(20), default="medium")  # low, medium, high, extreme
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_description = Column(String(500))
    affected_area = Column(JSON)  # Polygon coordinates
    estimated_size_hectares = Column(Float)
    
    # Fire behavior
    fire_behavior = Column(JSON)
    spread_rate = Column(String(20))  # slow, moderate, fast, extreme
    flame_height = Column(Float)
    spotting_distance = Column(Float)
    
    # Resources
    resources_deployed = Column(JSON)
    evacuation_zones = Column(JSON)
    road_closures = Column(JSON)
    
    # Timeline
    first_detected = Column(DateTime)
    reported_at = Column(DateTime)
    contained_at = Column(DateTime)
    controlled_at = Column(DateTime)
    extinguished_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("idx_incident_status", "status", "severity"),
        Index("idx_incident_location", "latitude", "longitude"),
        Index("idx_incident_time", "first_detected", "created_at"),
    )


class WeatherData(Base):
    """Weather data model for storing meteorological information."""
    
    __tablename__ = "weather_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    station_id = Column(String(50))
    
    # Weather parameters
    temperature = Column(Float)  # Celsius
    humidity = Column(Float)  # Percentage
    pressure = Column(Float)  # hPa
    wind_speed = Column(Float)  # km/h
    wind_direction = Column(Float)  # Degrees
    wind_gust = Column(Float)  # km/h
    precipitation = Column(Float)  # mm
    visibility = Column(Float)  # km
    cloud_cover = Column(Float)  # Percentage
    
    # Fire weather indices
    fire_weather_index = Column(Float)
    drought_code = Column(Float)
    buildup_index = Column(Float)
    fine_fuel_moisture = Column(Float)
    
    # Data source
    data_source = Column(String(100))  # api, sensor, manual
    quality_flag = Column(String(20), default="good")
    
    # Timestamps
    observation_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("idx_weather_location_time", "latitude", "longitude", "observation_time"),
        Index("idx_weather_fire_indices", "fire_weather_index", "drought_code"),
    )


class TelemetryData(Base):
    """Telemetry data model for storing drone sensor data."""
    
    __tablename__ = "telemetry_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Drone reference
    drone_id = Column(UUID(as_uuid=True), ForeignKey("drones.id"), nullable=False)
    
    # Position
    latitude = Column(Float)
    longitude = Column(Float)
    altitude = Column(Float)
    relative_altitude = Column(Float)
    
    # Attitude
    roll = Column(Float)
    pitch = Column(Float)
    yaw = Column(Float)
    
    # Velocity
    velocity_x = Column(Float)
    velocity_y = Column(Float)
    velocity_z = Column(Float)
    ground_speed = Column(Float)
    air_speed = Column(Float)
    
    # System status
    battery_voltage = Column(Float)
    battery_current = Column(Float)
    battery_remaining = Column(Float)
    flight_mode = Column(String(50))
    armed = Column(Boolean)
    
    # Sensors
    gps_fix_type = Column(Integer)
    gps_satellites = Column(Integer)
    vibration_x = Column(Float)
    vibration_y = Column(Float)
    vibration_z = Column(Float)
    
    # Environmental
    temperature = Column(Float)
    pressure = Column(Float)
    
    # Camera
    gimbal_roll = Column(Float)
    gimbal_pitch = Column(Float)
    gimbal_yaw = Column(Float)
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    drone = relationship("Drone", back_populates="telemetry")
    
    # Indexes
    __table_args__ = (
        Index("idx_telemetry_drone_time", "drone_id", "timestamp"),
        Index("idx_telemetry_location", "latitude", "longitude"),
    ) 