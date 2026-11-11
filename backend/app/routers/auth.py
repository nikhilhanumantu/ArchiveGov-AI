from fastapi import APIRouter, Response, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from backend.app.schemas.auth import LoginRequest, LoginResponse, ForgotPasswordRequest
from backend.app.services.auth_service import auth_service
from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user
from backend.app.core.security import verify_password, hash_password
from backend.app.database.repositories.user_repo import UserRepository
from backend.app.database.repositories.audit_repo import AuditRepository

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
user_repo = UserRepository()
audit_repo = AuditRepository()

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, response: Response):
    auth_result = auth_service.authenticate_user(
        username_or_email=request.username_or_email,
        password=request.password
    )
    # Set secure HttpOnly cookie
    response.set_cookie(
        key=settings.COOKIE_NAME,
        value=auth_result["access_token"],
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        samesite="lax",
        secure=False  # Allow local dev over http
    )
    return auth_result

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key=settings.COOKIE_NAME)
    return {"message": "Logged out successfully from sovereign session."}

@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    user_clean = {k: v for k, v in user.items() if k != "password_hash"}
    return user_clean

@router.post("/change-password")
def change_password(request: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    if not verify_password(request.current_password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Current password entered is incorrect.")
    
    if len(request.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")

    new_hash = hash_password(request.new_password)
    user_repo.update(user["_id"], {"password_hash": new_hash})

    audit_repo.log(
        user_id=user["_id"],
        action="PASSWORD_CHANGED",
        resource_type="user",
        resource_id=user["_id"]
    )
    return {"message": "Password updated successfully."}

@router.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest):
    return {
        "message": f"If an official account exists for {request.email}, password recovery credentials have been dispatched."
    }
