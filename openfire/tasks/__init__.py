"""
Background tasks and job processing for OpenWildfires platform.
"""

from openfire.tasks.celery_app import celery_app
from openfire.tasks.detection_tasks import (
    process_detection_task,
    batch_process_images_task,
    train_model_task
)
from openfire.tasks.alert_tasks import (
    send_alert_task,
    escalate_alert_task,
    send_notification_task
)
from openfire.tasks.drone_tasks import (
    monitor_drone_health_task,
    process_telemetry_task,
    execute_mission_task
)
from openfire.tasks.maintenance_tasks import (
    cleanup_old_data_task,
    backup_database_task,
    health_check_task
)

__all__ = [
    "celery_app",
    "process_detection_task",
    "batch_process_images_task", 
    "train_model_task",
    "send_alert_task",
    "escalate_alert_task",
    "send_notification_task",
    "monitor_drone_health_task",
    "process_telemetry_task",
    "execute_mission_task",
    "cleanup_old_data_task",
    "backup_database_task",
    "health_check_task"
] 