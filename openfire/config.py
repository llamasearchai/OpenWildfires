"""
Configuration management for OpenFire platform.
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://openfire:openfire@localhost:5432/openfire",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max connection overflow")
    
    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    max_connections: int = Field(default=20, description="Max Redis connections")
    
    class Config:
        env_prefix = "REDIS_"


class OpenAISettings(BaseSettings):
    """OpenAI API configuration settings."""
    
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(
        default="gpt-4-vision-preview",
        description="OpenAI model to use"
    )
    max_tokens: int = Field(default=1000, description="Max tokens per request")
    temperature: float = Field(default=0.1, description="Model temperature")
    
    class Config:
        env_prefix = "OPENAI_"


class DroneSettings(BaseSettings):
    """Drone configuration settings."""
    
    connection_string: str = Field(
        default="udp:127.0.0.1:14550",
        description="MAVLink connection string"
    )
    baud_rate: int = Field(default=57600, description="Serial baud rate")
    timeout: float = Field(default=30.0, description="Connection timeout")
    max_altitude: float = Field(default=120.0, description="Maximum flight altitude (m)")
    min_battery: float = Field(default=20.0, description="Minimum battery level (%)")
    
    class Config:
        env_prefix = "DRONE_"


class DetectionSettings(BaseSettings):
    """AI detection model settings."""
    
    model_path: str = Field(
        default="models/yolov8-fire-v2.pt",
        description="Path to detection model"
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Detection confidence threshold"
    )
    nms_threshold: float = Field(
        default=0.4,
        description="Non-maximum suppression threshold"
    )
    input_size: int = Field(default=640, description="Model input image size")
    device: str = Field(default="cuda", description="Inference device")
    
    class Config:
        env_prefix = "DETECTION_"


class AlertSettings(BaseSettings):
    """Alert system configuration."""
    
    # Twilio settings
    twilio_account_sid: Optional[str] = Field(default=None)
    twilio_auth_token: Optional[str] = Field(default=None)
    twilio_phone_number: Optional[str] = Field(default=None)
    
    # SendGrid settings
    sendgrid_api_key: Optional[str] = Field(default=None)
    sendgrid_from_email: Optional[str] = Field(default=None)
    
    # Emergency contacts
    emergency_phones: List[str] = Field(default_factory=list)
    emergency_emails: List[str] = Field(default_factory=list)
    
    # Alert thresholds
    fire_confidence_threshold: float = Field(default=0.8)
    smoke_confidence_threshold: float = Field(default=0.7)
    
    class Config:
        env_prefix = "ALERT_"


class APISettings(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-this",
        description="JWT secret key"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time"
    )
    
    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, description="Max log file size (bytes)")
    backup_count: int = Field(default=5, description="Number of backup log files")
    
    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    drone: DroneSettings = Field(default_factory=DroneSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings 