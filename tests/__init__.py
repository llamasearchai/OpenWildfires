"""
Test suite for OpenWildfires platform.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from openfire.database.models import Base
from openfire.config import get_settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create a test database engine."""
    # Use in-memory SQLite for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_settings():
    """Get test configuration settings."""
    return get_settings()


@pytest.fixture
def sample_image_data():
    """Provide sample image data for testing."""
    import numpy as np
    # Create a simple test image (RGB)
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection_data():
    """Provide sample detection data for testing."""
    return {
        "detection_type": "fire",
        "confidence": 0.85,
        "bounding_box": [100, 100, 200, 200],
        "latitude": 37.7749,
        "longitude": -122.4194,
        "altitude": 50.0
    }


@pytest.fixture
def sample_drone_data():
    """Provide sample drone data for testing."""
    return {
        "drone_id": "test_drone_001",
        "name": "Test Drone",
        "model": "DJI Mavic 3",
        "serial_number": "TEST123456",
        "max_altitude": 120.0,
        "max_speed": 15.0,
        "camera_resolution": "4K",
        "has_thermal_camera": True,
        "has_lidar": False
    }


@pytest.fixture
def sample_mission_data():
    """Provide sample mission data for testing."""
    return {
        "name": "Test Patrol Mission",
        "description": "Test mission for unit testing",
        "mission_type": "patrol",
        "waypoints": [
            [37.7749, -122.4194, 50.0],
            [37.7750, -122.4195, 50.0],
            [37.7751, -122.4196, 50.0]
        ],
        "flight_altitude": 50.0,
        "flight_speed": 5.0
    }


@pytest.fixture
def sample_weather_data():
    """Provide sample weather data for testing."""
    return {
        "temperature": 25.0,
        "humidity": 45.0,
        "wind_speed": 10.0,
        "wind_direction": 180.0,
        "precipitation": 0.0,
        "pressure": 1013.25
    } 