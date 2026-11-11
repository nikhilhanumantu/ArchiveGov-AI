from fastapi import APIRouter, Depends, HTTPException
from backend.app.core.dependencies import get_current_user
from backend.app.database.repositories.notification_repo import NotificationRepository

router = APIRouter(prefix="/api/v1/notifications", tags=["Notifications"])
notif_repo = NotificationRepository()

@router.get("")
def list_notifications(user: dict = Depends(get_current_user)):
    user_id = user["_id"]
    notifs = notif_repo.list_for_user(user_id)
    unread_count = notif_repo.count_unread(user_id)
    return {
        "notifications": notifs,
        "unread_count": unread_count
    }

@router.post("/{notif_id}/read")
def mark_read(notif_id: str, user: dict = Depends(get_current_user)):
    success = notif_repo.mark_as_read(notif_id)
    return {"success": success}

@router.post("/read-all")
def mark_all_read(user: dict = Depends(get_current_user)):
    count = notif_repo.mark_all_as_read(user["_id"])
    return {"marked_count": count}

