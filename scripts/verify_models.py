import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from backend.app.ai.enhancement.realesrgan_adapter import realesrgan_adapter
from backend.app.ai.enhancement.gfpgan_adapter import gfpgan_adapter
from backend.app.ai.inpainting.lama_adapter import lama_adapter
from backend.app.ai.ocr.paddleocr_adapter import paddleocr_adapter
from backend.app.database.mongodb import get_db_status

def verify():
    print("==================================================")
    print("ArchiveGov AI — System & AI Model Diagnostics")
    print("==================================================")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print("\n--- Model Adapters ---")
    print(f"Real-ESRGAN: {realesrgan_adapter.get_status()}")
    print(f"GFPGAN (Face Restoration): {gfpgan_adapter.get_status()}")
    print(f"LaMa Inpainting: {lama_adapter.get_status()}")
    print(f"PaddleOCR: {paddleocr_adapter.get_status()}")
    print("\n--- Database Status ---")
    print(get_db_status())
    print("==================================================")

if __name__ == "__main__":
    verify()

