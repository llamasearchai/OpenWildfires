"""
Database connection utilities for OpenWildfires platform.
"""

import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import structlog

from openfire.config import get_settings
from openfire.database.models import Base

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
    
    def get_engine(self):
        """Get synchronous database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.settings.database.url,
                echo=self.settings.database.echo,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        return self._engine
    
    def get_async_engine(self):
        """Get asynchronous database engine."""
        if self._async_engine is None:
            # Convert sync URL to async URL
            async_url = self.settings.database.url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            
            self._async_engine = create_async_engine(
                async_url,
                echo=self.settings.database.echo,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        return self._async_engine
    
    def get_session_factory(self):
        """Get synchronous session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.get_engine(),
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory
    
    def get_async_session_factory(self):
        """Get asynchronous session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.get_async_engine(),
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
        return self._async_session_factory
    
    def get_session(self) -> Session:
        """Get a synchronous database session."""
        session_factory = self.get_session_factory()
        return session_factory()
    
    async def get_async_session(self) -> AsyncSession:
        """Get an asynchronous database session."""
        async_session_factory = self.get_async_session_factory()
        return async_session_factory()
    
    async def create_tables(self):
        """Create all database tables."""
        try:
            async_engine = self.get_async_engine()
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables."""
        try:
            async_engine = self.get_async_engine()
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._engine:
            self._engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


def get_database_engine():
    """Get the database engine."""
    return db_manager.get_engine()


def get_async_database_engine():
    """Get the async database engine."""
    return db_manager.get_async_engine()


def get_database_session() -> Session:
    """Get a database session."""
    return db_manager.get_session()


async def get_async_database_session() -> AsyncSession:
    """Get an async database session."""
    return await db_manager.get_async_session()


async def get_database_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting database sessions."""
    async with db_manager.get_async_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    """Create all database tables."""
    await db_manager.create_tables()


async def drop_tables():
    """Drop all database tables."""
    await db_manager.drop_tables()


async def init_database():
    """Initialize the database with default data."""
    try:
        # Create tables
        await create_tables()
        
        # Create default admin user
        from openfire.database.models import User
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        async with db_manager.get_async_session_factory()() as session:
            # Check if admin user exists
            from sqlalchemy import select
            result = await session.execute(
                select(User).where(User.username == "admin")
            )
            admin_user = result.scalar_one_or_none()
            
            if not admin_user:
                # Create admin user
                admin_user = User(
                    username="admin",
                    email="admin@openwildfires.ai",
                    hashed_password=pwd_context.hash("admin123"),
                    full_name="System Administrator",
                    is_superuser=True,
                    is_active=True
                )
                session.add(admin_user)
                await session.commit()
                logger.info("Default admin user created")
            else:
                logger.info("Admin user already exists")
        
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def check_database_connection():
    """Check database connection health."""
    try:
        async with db_manager.get_async_session_factory()() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


class DatabaseHealthCheck:
    """Database health check utility."""
    
    @staticmethod
    async def check() -> dict:
        """Perform comprehensive database health check."""
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": asyncio.get_event_loop().time()
        }
        
        try:
            # Connection check
            connection_ok = await check_database_connection()
            health_status["checks"]["connection"] = {
                "status": "pass" if connection_ok else "fail",
                "message": "Database connection successful" if connection_ok else "Database connection failed"
            }
            
            # Table existence check
            try:
                async with db_manager.get_async_session_factory()() as session:
                    from sqlalchemy import text
                    result = await session.execute(
                        text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
                    )
                    table_count = result.scalar()
                    
                    health_status["checks"]["tables"] = {
                        "status": "pass" if table_count > 0 else "warn",
                        "message": f"Found {table_count} tables",
                        "table_count": table_count
                    }
            except Exception as e:
                health_status["checks"]["tables"] = {
                    "status": "fail",
                    "message": f"Table check failed: {e}"
                }
            
            # Performance check (simple query timing)
            try:
                import time
                start_time = time.time()
                async with db_manager.get_async_session_factory()() as session:
                    await session.execute("SELECT 1")
                query_time = (time.time() - start_time) * 1000  # ms
                
                health_status["checks"]["performance"] = {
                    "status": "pass" if query_time < 100 else "warn",
                    "message": f"Query response time: {query_time:.2f}ms",
                    "response_time_ms": query_time
                }
            except Exception as e:
                health_status["checks"]["performance"] = {
                    "status": "fail",
                    "message": f"Performance check failed: {e}"
                }
            
            # Overall status
            failed_checks = [
                check for check in health_status["checks"].values()
                if check["status"] == "fail"
            ]
            
            if failed_checks:
                health_status["status"] = "unhealthy"
            elif any(check["status"] == "warn" for check in health_status["checks"].values()):
                health_status["status"] = "degraded"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status 