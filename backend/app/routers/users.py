from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from backend.app.schemas.users import UserCreate, UserUpdate
from backend.app.core.dependencies import get_current_user, require_role
from backend.app.core.permissions import Role
from backend.app.core.security import hash_password
from backend.app.database.repositories.user_repo import UserRepository
from backend.app.database.repositories.audit_repo import AuditRepository

router = APIRouter(prefix="/api/v1/users", tags=["User Management"])
user_repo = UserRepository()
audit_repo = AuditRepository()

@router.get("")
def list_all_users(user: dict = Depends(require_role([Role.ADMIN]))):
    users = user_repo.list_users()
    # Strip password hashes
    cleaned = [{k: v for k, v in u.items() if k != "password_hash"} for u in users]
    return cleaned

@router.post("")
def create_user(user_in: UserCreate, admin_user: dict = Depends(require_role([Role.ADMIN]))):
    # Check duplicate email
    if user_repo.find_by_email(user_in.email):
        raise HTTPException(status_code=400, detail="User with this email already exists.")
    # Check duplicate employee ID
    if user_repo.find_by_employee_id(user_in.employee_id):
        raise HTTPException(status_code=400, detail="User with this Employee ID already exists.")

    new_user = {
        "name": user_in.name,
        "email": user_in.email.strip().lower(),
        "employee_id": user_in.employee_id.strip(),
        "password_hash": hash_password(user_in.password),
        "role": user_in.role,
        "department_id": user_in.department_id,
        "account_status": user_in.account_status,
        "created_by": admin_user["_id"]
    }
    created = user_repo.create(new_user)
    
    audit_repo.log(
        user_id=admin_user["_id"],
        action="USER_CREATED",
        resource_type="user",
        resource_id=created["_id"],
        metadata={"role": user_in.role, "email": user_in.email}
    )
    
    cleaned = {k: v for k, v in created.items() if k != "password_hash"}
    return cleaned

@router.put("/{user_id}")
def update_user(user_id: str, user_update: UserUpdate, admin_user: dict = Depends(require_role([Role.ADMIN]))):
    target = user_repo.find_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found.")

    update_dict = {k: v for k, v in user_update.dict().items() if v is not None}
    if "password" in update_dict:
        update_dict["password_hash"] = hash_password(update_dict.pop("password"))

    updated = user_repo.update(user_id, update_dict)
    
    audit_repo.log(
        user_id=admin_user["_id"],
        action="USER_UPDATED",
        resource_type="user",
        resource_id=user_id,
        metadata={"fields_updated": list(update_dict.keys())}
    )
    
    cleaned = {k: v for k, v in updated.items() if k != "password_hash"}
    return cleaned

@router.put("/{user_id}/status")
def toggle_user_status(user_id: str, new_status: str, admin_user: dict = Depends(require_role([Role.ADMIN]))):
    if new_status not in ["active", "suspended", "inactive"]:
        raise HTTPException(status_code=400, detail="Invalid status.")

    updated = user_repo.update(user_id, {"account_status": new_status})
    return {"message": f"User status set to {new_status}."}

