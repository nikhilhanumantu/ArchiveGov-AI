from pydantic import BaseModel, EmailStr, model_validator
from typing import Optional, Any

class LoginRequest(BaseModel):
    username_or_email: str  # Can be email or employee ID
    password: str

    @model_validator(mode="before")
    @classmethod
    def normalize_identifier(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # Accept username, email, identifier as aliases for username_or_email
            if "username_or_email" not in values or not values["username_or_email"]:
                for alias in ["username", "email", "identifier", "user"]:
                    if alias in values and values[alias]:
                        values["username_or_email"] = values[alias]
                        break
        return values

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    email: EmailStr
    new_password: str

