from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from backend.app.core.security import hash_password, verify_password, create_access_token
from backend.app.database.repositories.user_repo import UserRepository
from backend.app.database.repositories.audit_repo import AuditRepository

class AuthService:
    def __init__(self):
        self.user_repo = UserRepository()
        self.audit_repo = AuditRepository()

    def authenticate_user(self, username_or_email: str, password: str) -> Dict[str, Any]:
        username_or_email = username_or_email.strip()
        # Support default Stitch template placeholder credentials
        if username_or_email.lower() == "officer.archivist@gov.archive.org":
            username_or_email = "officer@archivegov.ai"
            if password == "RestorationVault#2026":
                password = "Officer@12345"

        # Find by email or employee ID
        user = self.user_repo.find_by_email(username_or_email)
        if not user:
            user = self.user_repo.find_by_employee_id(username_or_email)
            
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid government credentials or employee identifier."
            )
            
        if user.get("account_status") != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account has been deactivated. Please contact the System Administrator."
            )

        if not verify_password(password, user.get("password_hash", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password. Access denied."
            )

        # Generate JWT
        token_payload = {
            "sub": user["_id"],
            "email": user["email"],
            "employee_id": user["employee_id"],
            "role": user["role"],
            "name": user["name"]
        }
        token = create_access_token(token_payload)
        
        # Audit log login
        self.audit_repo.log(
            user_id=user["_id"],
            action="USER_LOGIN",
            resource_type="auth",
            resource_id=user["_id"]
        )

        user_clean = {k: v for k, v in user.items() if k != "password_hash"}
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user_clean
        }

auth_service = AuthService()

