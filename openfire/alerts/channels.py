"""
Notification channels for alert system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    async def send(self, alert_data: Dict[str, Any]) -> bool:
        """Send an alert through this channel."""
        pass


class SMSChannel(NotificationChannel):
    """SMS notification channel using Twilio."""
    
    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Twilio client."""
        try:
            from twilio.rest import Client
            self.client = Client(self.account_sid, self.auth_token)
            logger.info("Twilio SMS client initialized")
        except ImportError:
            logger.warning("Twilio not available - SMS notifications disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
    
    async def send(self, alert_data: Dict[str, Any]) -> bool:
        """Send SMS alert."""
        if not self.client:
            logger.warning("SMS client not available")
            return False
        
        try:
            message = self._format_sms_message(alert_data)
            recipients = self._get_sms_recipients(alert_data)
            
            results = []
            for recipient in recipients:
                try:
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_number,
                        to=recipient
                    )
                    results.append(True)
                    logger.info(f"SMS sent to {recipient}", message_sid=message_obj.sid)
                except Exception as e:
                    logger.error(f"Failed to send SMS to {recipient}: {e}")
                    results.append(False)
            
            return any(results)
            
        except Exception as e:
            logger.error(f"SMS sending failed: {e}")
            return False
    
    def _format_sms_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert data for SMS."""
        alert_type = alert_data.get('alert_type', 'Alert')
        priority = alert_data.get('priority', 'medium')
        message = alert_data.get('message', 'No details available')
        
        sms_text = f"🔥 OPENFIRE ALERT\n"
        sms_text += f"Type: {alert_type.upper()}\n"
        sms_text += f"Priority: {priority.upper()}\n"
        sms_text += f"Message: {message}\n"
        
        if alert_data.get('location'):
            lat, lon = alert_data['location'][:2]
            sms_text += f"Location: {lat:.4f}, {lon:.4f}\n"
        
        sms_text += f"Time: {alert_data.get('created_at', 'Unknown')}"
        
        # Truncate to SMS length limit
        return sms_text[:160]
    
    def _get_sms_recipients(self, alert_data: Dict[str, Any]) -> list:
        """Get SMS recipients based on alert data."""
        # In production, this would query a database or configuration
        # For now, return mock recipients
        return ["+1234567890"]  # Mock phone number


class EmailChannel(NotificationChannel):
    """Email notification channel using SendGrid."""
    
    def __init__(
        self,
        api_key: str,
        from_email: str
    ):
        self.api_key = api_key
        self.from_email = from_email
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize SendGrid client."""
        try:
            import sendgrid
            self.client = sendgrid.SendGridAPIClient(api_key=self.api_key)
            logger.info("SendGrid email client initialized")
        except ImportError:
            logger.warning("SendGrid not available - email notifications disabled")
        except Exception as e:
            logger.error(f"Failed to initialize SendGrid client: {e}")
    
    async def send(self, alert_data: Dict[str, Any]) -> bool:
        """Send email alert."""
        if not self.client:
            logger.warning("Email client not available")
            return False
        
        try:
            from sendgrid.helpers.mail import Mail
            
            subject, html_content = self._format_email_content(alert_data)
            recipients = self._get_email_recipients(alert_data)
            
            results = []
            for recipient in recipients:
                try:
                    message = Mail(
                        from_email=self.from_email,
                        to_emails=recipient,
                        subject=subject,
                        html_content=html_content
                    )
                    
                    response = self.client.send(message)
                    results.append(response.status_code == 202)
                    logger.info(f"Email sent to {recipient}", status=response.status_code)
                    
                except Exception as e:
                    logger.error(f"Failed to send email to {recipient}: {e}")
                    results.append(False)
            
            return any(results)
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return False
    
    def _format_email_content(self, alert_data: Dict[str, Any]) -> tuple:
        """Format alert data for email."""
        alert_type = alert_data.get('alert_type', 'Alert')
        priority = alert_data.get('priority', 'medium')
        message = alert_data.get('message', 'No details available')
        
        # Subject
        subject = f"🔥 OpenFire Alert: {alert_type} ({priority.upper()})"
        
        # HTML content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #ff4444; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
                .priority-{priority} {{ border-left: 5px solid {"#ff0000" if priority == "critical" else "#ff8800" if priority == "high" else "#ffaa00" if priority == "medium" else "#00aa00"}; }}
                .metadata {{ background-color: #f5f5f5; padding: 10px; margin-top: 15px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🔥 OpenFire Wildfire Detection Alert</h2>
            </div>
            <div class="content priority-{priority}">
                <h3>Alert Details</h3>
                <p><strong>Type:</strong> {alert_type}</p>
                <p><strong>Priority:</strong> {priority.upper()}</p>
                <p><strong>Message:</strong> {message}</p>
                <p><strong>Time:</strong> {alert_data.get('created_at', 'Unknown')}</p>
        """
        
        if alert_data.get('location'):
            lat, lon = alert_data['location'][:2]
            html_content += f"""
                <p><strong>Location:</strong> {lat:.6f}, {lon:.6f}</p>
                <p><a href="https://maps.google.com/?q={lat},{lon}" target="_blank">View on Google Maps</a></p>
            """
        
        if alert_data.get('metadata'):
            html_content += f"""
                <div class="metadata">
                    <h4>Additional Information</h4>
                    <pre>{str(alert_data['metadata'])}</pre>
                </div>
            """
        
        html_content += """
            </div>
            <p><em>This is an automated alert from the OpenFire wildfire detection system.</em></p>
        </body>
        </html>
        """
        
        return subject, html_content
    
    def _get_email_recipients(self, alert_data: Dict[str, Any]) -> list:
        """Get email recipients based on alert data."""
        # In production, this would query a database or configuration
        # For now, return mock recipients
        return ["admin@example.com"]  # Mock email


class PushChannel(NotificationChannel):
    """Push notification channel."""
    
    def __init__(self):
        self.subscribers = []  # Mock subscriber list
        logger.info("Push notification channel initialized")
    
    async def send(self, alert_data: Dict[str, Any]) -> bool:
        """Send push notification."""
        try:
            notification = self._format_push_notification(alert_data)
            
            # Mock push notification sending
            logger.info(
                "Push notification sent",
                title=notification['title'],
                body=notification['body'],
                subscribers=len(self.subscribers)
            )
            
            # In production, this would use a service like Firebase Cloud Messaging
            # or Apple Push Notification Service
            
            return True
            
        except Exception as e:
            logger.error(f"Push notification failed: {e}")
            return False
    
    def _format_push_notification(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """Format alert data for push notification."""
        alert_type = alert_data.get('alert_type', 'Alert')
        priority = alert_data.get('priority', 'medium')
        message = alert_data.get('message', 'No details available')
        
        # Priority emoji
        priority_emoji = {
            'critical': '🚨',
            'high': '🔥',
            'medium': '⚠️',
            'low': 'ℹ️'
        }.get(priority, 'ℹ️')
        
        return {
            'title': f"{priority_emoji} OpenFire Alert",
            'body': f"{alert_type}: {message}",
            'data': {
                'alert_id': alert_data.get('alert_id'),
                'priority': priority,
                'type': alert_type,
                'location': alert_data.get('location')
            }
        }
    
    def subscribe(self, device_token: str) -> bool:
        """Subscribe a device to push notifications."""
        if device_token not in self.subscribers:
            self.subscribers.append(device_token)
            logger.info(f"Device subscribed to push notifications: {device_token[:10]}...")
            return True
        return False
    
    def unsubscribe(self, device_token: str) -> bool:
        """Unsubscribe a device from push notifications."""
        if device_token in self.subscribers:
            self.subscribers.remove(device_token)
            logger.info(f"Device unsubscribed from push notifications: {device_token[:10]}...")
            return True
        return False


class WebhookChannel(NotificationChannel):
    """Webhook notification channel for external integrations."""
    
    def __init__(self, webhook_url: str, secret: Optional[str] = None):
        self.webhook_url = webhook_url
        self.secret = secret
        logger.info(f"Webhook channel initialized: {webhook_url}")
    
    async def send(self, alert_data: Dict[str, Any]) -> bool:
        """Send webhook notification."""
        try:
            import httpx
            import hashlib
            import hmac
            import json
            
            payload = {
                'event': 'alert',
                'data': alert_data,
                'timestamp': alert_data.get('created_at')
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'OpenFire-Alert-System/1.0'
            }
            
            # Add signature if secret is provided
            if self.secret:
                payload_str = json.dumps(payload, sort_keys=True)
                signature = hmac.new(
                    self.secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers['X-OpenFire-Signature'] = f"sha256={signature}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=10.0
                )
                
                success = response.status_code == 200
                logger.info(
                    f"Webhook notification sent",
                    url=self.webhook_url,
                    status=response.status_code,
                    success=success
                )
                
                return success
                
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False 