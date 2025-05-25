"""
Celery application configuration for OpenWildfires platform.
"""

from celery import Celery
from celery.schedules import crontab
import structlog

from openfire.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Create Celery app
celery_app = Celery(
    "openfire",
    broker=settings.redis.url,
    backend=settings.redis.url,
    include=[
        "openfire.tasks.detection_tasks",
        "openfire.tasks.alert_tasks", 
        "openfire.tasks.drone_tasks",
        "openfire.tasks.maintenance_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "openfire.tasks.detection_tasks.*": {"queue": "detection"},
        "openfire.tasks.alert_tasks.*": {"queue": "alerts"},
        "openfire.tasks.drone_tasks.*": {"queue": "drones"},
        "openfire.tasks.maintenance_tasks.*": {"queue": "maintenance"},
    },
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Result backend settings
    result_expires=3600,
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    
    # Task execution settings
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    task_reject_on_worker_lost=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        # Health monitoring every 5 minutes
        "health-check": {
            "task": "openfire.tasks.maintenance_tasks.health_check_task",
            "schedule": crontab(minute="*/5"),
        },
        
        # Drone health monitoring every minute
        "drone-health-monitor": {
            "task": "openfire.tasks.drone_tasks.monitor_drone_health_task",
            "schedule": crontab(minute="*"),
        },
        
        # Alert escalation check every 2 minutes
        "alert-escalation": {
            "task": "openfire.tasks.alert_tasks.escalate_alert_task",
            "schedule": crontab(minute="*/2"),
        },
        
        # Data cleanup daily at 2 AM
        "cleanup-old-data": {
            "task": "openfire.tasks.maintenance_tasks.cleanup_old_data_task",
            "schedule": crontab(hour=2, minute=0),
        },
        
        # Database backup daily at 3 AM
        "backup-database": {
            "task": "openfire.tasks.maintenance_tasks.backup_database_task",
            "schedule": crontab(hour=3, minute=0),
        },
        
        # Model retraining weekly on Sunday at 4 AM
        "retrain-models": {
            "task": "openfire.tasks.detection_tasks.train_model_task",
            "schedule": crontab(hour=4, minute=0, day_of_week=0),
        },
    },
)

# Task annotations for monitoring
celery_app.conf.task_annotations = {
    "*": {
        "rate_limit": "100/m",
    },
    "openfire.tasks.detection_tasks.process_detection_task": {
        "rate_limit": "50/m",
        "time_limit": 120,
    },
    "openfire.tasks.alert_tasks.send_alert_task": {
        "rate_limit": "200/m",
        "time_limit": 30,
    },
    "openfire.tasks.drone_tasks.process_telemetry_task": {
        "rate_limit": "1000/m",
        "time_limit": 10,
    },
}


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup."""
    logger.info(f"Request: {self.request!r}")
    return "Celery is working!"


# Error handling
@celery_app.task(bind=True)
def error_handler(self, uuid, err, traceback):
    """Handle task errors."""
    logger.error(
        "Task failed",
        task_id=uuid,
        error=str(err),
        traceback=traceback
    )


# Task success callback
@celery_app.task(bind=True)
def success_handler(self, retval, task_id, args, kwargs):
    """Handle successful task completion."""
    logger.info(
        "Task completed successfully",
        task_id=task_id,
        result=retval
    )


# Configure error handling
celery_app.conf.task_annotations["*"]["on_failure"] = error_handler
celery_app.conf.task_annotations["*"]["on_success"] = success_handler 