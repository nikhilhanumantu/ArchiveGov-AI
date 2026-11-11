from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class DocumentMetadataInput(BaseModel):
    title: str
    department_id: str = "Land Revenue & Settlement"
    category: str = "Historical Land Records"
    original_date: str = "1912"
    language: str = "English & Tamil Vernacular"
    archive_reference_id: str
    description: Optional[str] = None
    assigned_reviewer: Optional[str] = None

class DocumentUpdateInput(BaseModel):
    title: Optional[str] = None
    department_id: Optional[str] = None
    category: Optional[str] = None
    original_date: Optional[str] = None
    language: Optional[str] = None
    archive_reference_id: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    assigned_reviewer: Optional[str] = None

