from typing import Optional, List
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.app.core.security import decode_access_token
from backend.app.core.config import settings
from backend.app.core.permissions import Role, Permission, has_permission
from backend.app.database.repositories.user_repo import UserRepository

security_bearer = HTTPBearer(auto_error=False)
user_repo = UserRepository()

def get_current_user_optional(
    request: Request,
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)
) -> Optional[dict]:
    token = None
    if bearer:
        token = bearer.credentials
    if not token and settings.COOKIE_NAME in request.cookies:
        token = request.cookies[settings.COOKIE_NAME]
        
    if not token:
        return None
        
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None
        
    user = user_repo.find_by_id(payload["sub"])
    return user

def get_current_user(
    request: Request,
    user: Optional[dict] = Depends(get_current_user_optional)
) -> dict:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in with government credentials."
        )
    return user

def require_role(allowed_roles: List[Role]):
    def role_checker(user: dict = Depends(get_current_user)) -> dict:
        user_role = user.get("role")
        if user_role not in [r.value for r in allowed_roles] and user_role != Role.ADMIN.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: requires one of {allowed_roles} clearance."
            )
        return user
    return role_checker

def require_permission(perm: Permission):
    def perm_checker(user: dict = Depends(get_current_user)) -> dict:
        user_role = user.get("role", "")
        if not has_permission(user_role, perm):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: insufficient permission '{perm.value}'."
            )
        return user
    return perm_checker

