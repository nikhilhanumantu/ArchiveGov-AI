import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from backend.app.database.mongodb import get_db, sync_disk_if_mock
from backend.app.core.security import hash_password

def create_admin(name="Dr. Rajeshwar Sharma", email="admin@archivegov.ai", password="Admin@12345", employee_id="GOV-ADM-001"):
    db = get_db()
    existing = db.users.find_one({"email": email.lower()})
    if existing:
        db.users.update_one(
            {"_id": existing["_id"]},
            {"$set": {
                "password_hash": hash_password(password),
                "role": "admin",
                "account_status": "active"
            }}
        )
        print(f"Updated existing admin user: {email}")
    else:
        db.users.insert_one({
            "name": name,
            "email": email.lower(),
            "employee_id": employee_id,
            "password_hash": hash_password(password),
            "role": "admin",
            "department_id": "Central Government National Archives",
            "account_status": "active"
        })
        print(f"Created new admin user: {email}")
    sync_disk_if_mock()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or reset ArchiveGov AI Administrator")
    parser.add_argument("--name", default="Dr. Rajeshwar Sharma")
    parser.add_argument("--email", default="admin@archivegov.ai")
    parser.add_argument("--password", default="Admin@12345")
    parser.add_argument("--employee-id", default="GOV-ADM-001")
    args = parser.parse_args()
    create_admin(args.name, args.email, args.password, args.employee_id)

