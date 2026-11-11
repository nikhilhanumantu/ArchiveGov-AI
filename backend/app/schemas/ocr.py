from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class OCRSaveRequest(BaseModel):
    document_id: str
    corrected_text: str
    notes: Optional[str] = None

