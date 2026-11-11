from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    employee_id: str
    password: str
    role: str = "archive_officer"
    department_id: str = "Central Government Archives"
    account_status: str = "active"

class UserUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    department_id: Optional[str] = None
    account_status: Optional[str] = None
    password: Optional[str] = None

