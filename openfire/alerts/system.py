"""
Comprehensive alert and notification system.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import structlog

from openfire.config import get_settings
from openfire.alerts.channels import SMSChannel, EmailChannel, PushChannel

logger = structlog.get_logger(__name__)


class AlertSystem:
    """Advanced alert and notification system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.channels = {}
        self.alert_history = []
        self._initialize_channels()
    
    def _initialize_channels(self):
        """Initialize notification channels."""
        try:
            # SMS Channel
            if self.settings.alerts.twilio_account_sid:
                self.channels['sms'] = SMSChannel(
                    account_sid=self.settings.alerts.twilio_account_sid,
                    auth_token=self.settings.alerts.twilio_auth_token,
                    from_number=self.settings.alerts.twilio_phone_number
                )
                logger.info("SMS channel initialized")
            
            # Email Channel
            if self.settings.alerts.sendgrid_api_key:
                self.channels['email'] = EmailChannel(
                    api_key=self.settings.alerts.sendgrid_api_key,
                    from_email=self.settings.alerts.sendgrid_from_email
                )
                logger.info("Email channel initialized")
            
            # Push Notification Channel
            self.channels['push'] = PushChannel()
            logger.info("Push notification channel initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize alert channels: {e}")
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send an alert through appropriate channels."""
        try:
            alert_id = alert_data.get('alert_id', 'unknown')
            priority = alert_data.get('priority', 'medium')
            alert_type = alert_data.get('alert_type', 'general')
            
            logger.info(f"Sending alert {alert_id}", priority=priority, type=alert_type)
            
            # Determine channels based on priority and type
            channels_to_use = self._select_channels(priority, alert_type)
            
            # Send through selected channels
            results = []
            for channel_name in channels_to_use:
                if channel_name in self.channels:
                    try:
                        result = await self.channels[channel_name].send(alert_data)
                        results.append(result)
                        logger.info(f"Alert sent via {channel_name}", success=result)
                    except Exception as e:
                        logger.error(f"Failed to send alert via {channel_name}: {e}")
                        results.append(False)
            
            # Store in history
            alert_record = {
                **alert_data,
                'sent_at': datetime.utcnow().isoformat(),
                'channels_used': channels_to_use,
                'success': any(results)
            }
            self.alert_history.append(alert_record)
            
            return any(results)
            
        except Exception as e:
            logger.error(f"Alert sending failed: {e}")
            return False
    
    async def send_emergency_alert(
        self,
        message: str,
        location: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a high-priority emergency alert."""
        emergency_alert = {
            'alert_id': f"emergency_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'alert_type': 'emergency',
            'priority': 'critical',
            'message': message,
            'location': location,
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat()
        }
        
        return await self.send_alert(emergency_alert)
    
    async def send_fire_detection_alert(
        self,
        detection_data: Dict[str, Any],
        confidence: float,
        location: Optional[List[float]] = None
    ) -> bool:
        """Send a fire detection alert."""
        priority = self._determine_priority(confidence, detection_data)
        
        fire_alert = {
            'alert_id': f"fire_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'alert_type': 'wildfire_detection',
            'priority': priority,
            'message': f"Wildfire detected with {confidence:.1%} confidence",
            'location': location,
            'metadata': {
                'detection_data': detection_data,
                'confidence': confidence,
                'detection_time': datetime.utcnow().isoformat()
            },
            'created_at': datetime.utcnow().isoformat()
        }
        
        return await self.send_alert(fire_alert)
    
    def _select_channels(self, priority: str, alert_type: str) -> List[str]:
        """Select appropriate channels based on priority and type."""
        channels = []
        
        if priority in ['critical', 'high']:
            # High priority alerts use all available channels
            channels.extend(['sms', 'email', 'push'])
        elif priority == 'medium':
            # Medium priority uses email and push
            channels.extend(['email', 'push'])
        else:
            # Low priority uses push only
            channels.append('push')
        
        # Emergency alerts always use all channels
        if alert_type == 'emergency':
            channels = ['sms', 'email', 'push']
        
        # Filter to only available channels
        return [ch for ch in channels if ch in self.channels]
    
    def _determine_priority(self, confidence: float, detection_data: Dict[str, Any]) -> str:
        """Determine alert priority based on confidence and detection data."""
        if confidence >= 0.9:
            return 'critical'
        elif confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-limit:]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return [alert for alert in self.alert_history 
                if alert.get('status') == 'active'] 