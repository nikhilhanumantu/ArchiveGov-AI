from enum import Enum
from typing import List, Dict, Set

class Role(str, Enum):
    ADMIN = "admin"
    ARCHIVE_OFFICER = "archive_officer"
    DATA_OPERATOR = "data_operator"
    REVIEWER = "reviewer"

class Permission(str, Enum):
    # Documents
    DOCUMENT_VIEW = "document:view"
    DOCUMENT_UPLOAD = "document:upload"
    DOCUMENT_EDIT = "document:edit"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_DOWNLOAD = "document:download"
    
    # Restoration
    RESTORATION_START = "restoration:start"
    RESTORATION_INPAINT = "restoration:inpaint"
    RESTORATION_COMPARE = "restoration:compare"
    
    # OCR
    OCR_RUN = "ocr:run"
    OCR_EDIT = "ocr:edit"
    OCR_EXPORT = "ocr:export"
    
    # Review
    REVIEW_SUBMIT = "review:submit"
    REVIEW_APPROVE = "review:approve"
    REVIEW_REJECT = "review:reject"
    
    # Administration
    USERS_MANAGE = "users:manage"
    SETTINGS_EDIT = "settings:edit"
    AUDIT_VIEW = "audit:view"
    REPORTS_VIEW = "reports:view"

ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    
    Role.ARCHIVE_OFFICER: {
        Permission.DOCUMENT_VIEW,
        Permission.DOCUMENT_UPLOAD,
        Permission.DOCUMENT_EDIT,
        Permission.DOCUMENT_DOWNLOAD,
        Permission.RESTORATION_START,
        Permission.RESTORATION_INPAINT,
        Permission.RESTORATION_COMPARE,
        Permission.OCR_RUN,
        Permission.OCR_EDIT,
        Permission.OCR_EXPORT,
        Permission.REPORTS_VIEW,
    },
    
    Role.DATA_OPERATOR: {
        Permission.DOCUMENT_VIEW,
        Permission.DOCUMENT_UPLOAD,
        Permission.DOCUMENT_DOWNLOAD,
        Permission.OCR_RUN,
        Permission.OCR_EDIT,
        Permission.OCR_EXPORT,
    },
    
    Role.REVIEWER: {
        Permission.DOCUMENT_VIEW,
        Permission.DOCUMENT_DOWNLOAD,
        Permission.RESTORATION_COMPARE,
        Permission.REVIEW_SUBMIT,
        Permission.REVIEW_APPROVE,
        Permission.REVIEW_REJECT,
        Permission.OCR_EXPORT,
        Permission.REPORTS_VIEW,
    }
}

def has_permission(user_role: str, permission: Permission) -> bool:
    """Check if a given role has the requested permission."""
    try:
        role_enum = Role(user_role)
        return permission in ROLE_PERMISSIONS.get(role_enum, set())
    except ValueError:
        return False

