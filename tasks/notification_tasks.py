"""
Notification Tasks
Tasks for sending notifications and alerts
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from celery_app import app, DocumentTask, get_db_session
from models import Document, User, ProcessingJob

logger = logging.getLogger(__name__)


# Email configuration
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "noreply@documentai.com")

# Webhook configuration
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


@app.task(bind=True, base=DocumentTask, name="tasks.notification_tasks.send_processing_complete")
def send_processing_complete(self, document_id: str) -> Dict:
    """
    Send notification when document processing is complete
    
    Args:
        document_id: Document UUID
        
    Returns:
        Notification status
    """
    try:
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            user = session.query(User).filter_by(id=document.user_id).first()
            if not user:
                raise ValueError(f"User {document.user_id} not found")
        
        # Send email notification
        if user.email:
            email_result = _send_email(
                to=user.email,
                subject=f"Document Processing Complete - {document.original_filename}",
                body=f"""
                Your document has been processed successfully!
                
                Document: {document.original_filename}
                Processing Time: {document.processing_time_ms}ms
                Detection Count: {document.metadata.get('detection_count', 0)}
                
                You can view the results at: {os.getenv('APP_URL', 'http://localhost:8000')}/documents/{document_id}
                
                Best regards,
                Document AI Team
                """
            )
        else:
            email_result = {"sent": False, "reason": "No email address"}
        
        # Send webhook notification
        webhook_result = _send_webhook({
            "event": "document.processing.complete",
            "document_id": str(document_id),
            "user_id": str(user.id),
            "filename": document.original_filename,
            "status": document.status,
            "processing_time_ms": document.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Send Slack notification if configured
        slack_result = _send_slack_notification(
            f"✅ Document processed: *{document.original_filename}*\n"
            f"User: {user.username}\n"
            f"Processing time: {document.processing_time_ms}ms"
        )
        
        return {
            "document_id": document_id,
            "email": email_result,
            "webhook": webhook_result,
            "slack": slack_result
        }
        
    except Exception as e:
        logger.error(f"Failed to send processing complete notification: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.notification_tasks.send_failure_notification")
def send_failure_notification(self, task_name: str, task_id: str, error: str) -> Dict:
    """
    Send notification when a task fails
    
    Args:
        task_name: Name of the failed task
        task_id: Task UUID
        error: Error message
        
    Returns:
        Notification status
    """
    try:
        # Send to administrators
        admin_emails = os.getenv("ADMIN_EMAILS", "").split(",")
        
        email_results = []
        for email in admin_emails:
            if email.strip():
                result = _send_email(
                    to=email.strip(),
                    subject=f"Task Failure Alert - {task_name}",
                    body=f"""
                    A task has failed and requires attention.
                    
                    Task Name: {task_name}
                    Task ID: {task_id}
                    Error: {error}
                    Timestamp: {datetime.utcnow().isoformat()}
                    
                    Please check the logs for more details.
                    """
                )
                email_results.append(result)
        
        # Send Slack alert
        slack_result = _send_slack_notification(
            f"🚨 *Task Failure Alert*\n"
            f"Task: `{task_name}`\n"
            f"ID: `{task_id}`\n"
            f"Error: ```{error}```",
            color="danger"
        )
        
        # Log to monitoring system
        _send_to_monitoring({
            "event": "task.failure",
            "task_name": task_name,
            "task_id": task_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "task_name": task_name,
            "task_id": task_id,
            "email_results": email_results,
            "slack": slack_result
        }
        
    except Exception as e:
        logger.error(f"Failed to send failure notification: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.notification_tasks.send_quota_alert")
def send_quota_alert(self, user_id: str, quota_type: str, usage: int, limit: int) -> Dict:
    """
    Send quota alert to user
    
    Args:
        user_id: User UUID
        quota_type: Type of quota (storage, api_calls, etc.)
        usage: Current usage
        limit: Quota limit
        
    Returns:
        Notification status
    """
    try:
        with get_db_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                raise ValueError(f"User {user_id} not found")
        
        usage_percentage = (usage / limit) * 100
        
        # Send email
        email_result = _send_email(
            to=user.email,
            subject=f"Quota Alert - {quota_type.replace('_', ' ').title()}",
            body=f"""
            You are approaching your {quota_type.replace('_', ' ')} limit.
            
            Current Usage: {usage} / {limit} ({usage_percentage:.1f}%)
            
            Please consider upgrading your plan or reducing usage.
            
            View your usage: {os.getenv('APP_URL', 'http://localhost:8000')}/account/usage
            """
        )
        
        # Send in-app notification
        _create_in_app_notification(
            user_id=user_id,
            type="quota_alert",
            title=f"{quota_type.replace('_', ' ').title()} Quota Alert",
            message=f"You have used {usage_percentage:.1f}% of your {quota_type.replace('_', ' ')} quota.",
            data={
                "quota_type": quota_type,
                "usage": usage,
                "limit": limit,
                "percentage": usage_percentage
            }
        )
        
        return {
            "user_id": user_id,
            "quota_type": quota_type,
            "usage_percentage": usage_percentage,
            "email": email_result
        }
        
    except Exception as e:
        logger.error(f"Failed to send quota alert: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.notification_tasks.send_daily_summary")
def send_daily_summary(self, user_id: str) -> Dict:
    """
    Send daily summary email to user
    
    Args:
        user_id: User UUID
        
    Returns:
        Notification status
    """
    try:
        with get_db_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            # Get user's activity for the day
            from datetime import datetime, timedelta
            today = datetime.utcnow().date()
            yesterday = today - timedelta(days=1)
            
            documents_processed = session.query(Document).filter(
                Document.user_id == user_id,
                Document.created_at >= yesterday,
                Document.created_at < today,
                Document.status == 'completed'
            ).count()
            
            total_detections = session.query(Document).filter(
                Document.user_id == user_id,
                Document.created_at >= yesterday,
                Document.created_at < today
            ).join(Document.detection_results).count()
        
        if documents_processed == 0:
            # Don't send summary if no activity
            return {
                "user_id": user_id,
                "sent": False,
                "reason": "No activity"
            }
        
        # Send summary email
        email_result = _send_email(
            to=user.email,
            subject="Your Daily Document AI Summary",
            body=f"""
            Here's your activity summary for {yesterday.strftime('%B %d, %Y')}:
            
            📄 Documents Processed: {documents_processed}
            🔍 Total Detections: {total_detections}
            ⚡ Average Processing Time: {_get_avg_processing_time(user_id, yesterday, today)}ms
            
            Top Document Types:
            {_get_top_document_types(user_id, yesterday, today)}
            
            View detailed analytics: {os.getenv('APP_URL', 'http://localhost:8000')}/analytics
            
            Keep up the great work!
            
            Best regards,
            Document AI Team
            """
        )
        
        return {
            "user_id": user_id,
            "documents_processed": documents_processed,
            "email": email_result
        }
        
    except Exception as e:
        logger.error(f"Failed to send daily summary: {e}")
        raise


# Helper functions
def _send_email(to: str, subject: str, body: str, html: Optional[str] = None) -> Dict:
    """Send email using SMTP"""
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SMTP_FROM
        msg['To'] = to
        
        # Add text part
        text_part = MIMEText(body, 'plain')
        msg.attach(text_part)
        
        # Add HTML part if provided
        if html:
            html_part = MIMEText(html, 'html')
            msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USER and SMTP_PASSWORD:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
            
            server.send_message(msg)
        
        return {"sent": True, "to": to}
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return {"sent": False, "error": str(e)}


def _send_webhook(data: Dict) -> Dict:
    """Send webhook notification"""
    if not WEBHOOK_URL:
        return {"sent": False, "reason": "No webhook URL configured"}
    
    try:
        response = requests.post(
            WEBHOOK_URL,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        
        return {
            "sent": True,
            "status_code": response.status_code,
            "response": response.text
        }
        
    except Exception as e:
        logger.error(f"Failed to send webhook: {e}")
        return {"sent": False, "error": str(e)}


def _send_slack_notification(text: str, color: str = "good") -> Dict:
    """Send Slack notification"""
    if not SLACK_WEBHOOK_URL:
        return {"sent": False, "reason": "No Slack webhook configured"}
    
    try:
        payload = {
            "attachments": [
                {
                    "color": color,
                    "text": text,
                    "mrkdwn_in": ["text"],
                    "footer": "Document AI",
                    "ts": int(datetime.utcnow().timestamp())
                }
            ]
        }
        
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        return {"sent": True}
        
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")
        return {"sent": False, "error": str(e)}


def _send_to_monitoring(data: Dict):
    """Send event to monitoring system"""
    # This would integrate with your monitoring system (e.g., Datadog, New Relic)
    logger.info(f"Monitoring event: {data}")


def _create_in_app_notification(user_id: str, type: str, title: str, 
                              message: str, data: Optional[Dict] = None):
    """Create in-app notification"""
    # This would store notification in database for in-app display
    logger.info(f"In-app notification for {user_id}: {title}")


def _get_avg_processing_time(user_id: str, start_date: datetime, end_date: datetime) -> float:
    """Calculate average processing time for user's documents"""
    with get_db_session() as session:
        avg_time = session.query(
            func.avg(Document.processing_time_ms)
        ).filter(
            Document.user_id == user_id,
            Document.created_at >= start_date,
            Document.created_at < end_date,
            Document.processing_time_ms.isnot(None)
        ).scalar()
        
        return round(avg_time or 0, 2)


def _get_top_document_types(user_id: str, start_date: datetime, end_date: datetime) -> str:
    """Get top document types for user"""
    with get_db_session() as session:
        from sqlalchemy import func
        
        results = session.query(
            Document.document_type,
            func.count(Document.id).label('count')
        ).filter(
            Document.user_id == user_id,
            Document.created_at >= start_date,
            Document.created_at < end_date
        ).group_by(Document.document_type).order_by(
            func.count(Document.id).desc()
        ).limit(3).all()
        
        if not results:
            return "No documents processed"
        
        lines = []
        for doc_type, count in results:
            lines.append(f"  • {doc_type}: {count}")
        
        return "\n".join(lines)


# Scheduled tasks
@app.task(name="tasks.notification_tasks.send_daily_summaries")
def send_daily_summaries():
    """Send daily summaries to all active users"""
    with get_db_session() as session:
        active_users = session.query(User).filter(
            User.is_active == True,
            User.metadata['notifications']['daily_summary'] == True
        ).all()
        
        for user in active_users:
            send_daily_summary.delay(str(user.id))
    
    return {
        "users_notified": len(active_users)
    }