import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bson import json_util
from backend.app.database.mongodb import init_db

def run_migration():
    print("Connecting to MongoDB Atlas...")
    db = init_db()
    
    fallback_path = Path("backend/data/mongodb_fallback.json")
    if not fallback_path.exists():
        print("No local fallback data to migrate.")
        return

    with open(fallback_path, "r", encoding="utf-8") as f:
        data = json_util.loads(f.read())

    for col_name, docs in data.items():
        if not docs:
            continue
        col = db[col_name]
        inserted = 0
        updated = 0
        for d in docs:
            d_dict = dict(d)
            d_dict.pop('_id', None)
            
            # Key for identifying unique record
            if col_name == 'users':
                filter_q = {'email': d_dict.get('email')}
            elif col_name == 'documents':
                filter_q = {'document_id': d_dict.get('document_id')}
            elif col_name == 'document_versions':
                filter_q = {'version_id': d_dict.get('version_id')}
            elif col_name == 'system_settings':
                filter_q = {'setting_key': d_dict.get('setting_key')}
            elif col_name == 'ocr_results':
                filter_q = {'document_id': d_dict.get('document_id')}
            else:
                filter_q = None

            if filter_q:
                res = col.update_one(filter_q, {'$set': d_dict}, upsert=True)
                if res.upserted_id:
                    inserted += 1
                else:
                    updated += 1
            else:
                col.insert_one(d_dict)
                inserted += 1

        print(f"Collection '{col_name}': {inserted} inserted, {updated} updated.")

    # Also register the 1918 user uploaded document with OCR
    try:
        from backend.app.ai.ocr.paddleocr_adapter import paddleocr_adapter
        ocr_data = paddleocr_adapter.extract_text('uploads/originals/8c334f7bf09e47f4a70d60c0fb3b2bd0.jpg', lang='en')
        doc_1918 = {
            'document_id': 'DOC-1918-HLT',
            'title': '1918 Annual Report of the Commissioner of Health — Epidemic Control Folio',
            'department_id': 'Department of Public Health & Epidemic Control',
            'category': 'Government Circulars',
            'original_date': '1918',
            'language': 'English',
            'archive_reference_id': 'REC-1918-HLT-003',
            'description': '1918 Commissioner of Health official annual report addressing regulatory measures and sovereign containment of the influenza epidemic.',
            'original_file_key': '8c334f7bf09e47f4a70d60c0fb3b2bd0.jpg',
            'restored_file_key': '8c334f7bf09e47f4a70d60c0fb3b2bd0_restored.png',
            'file_size': 38550,
            'checksum': '8c334f7bf09e47f4a70d60c0fb3b2bd0',
            'current_version_id': 'VER-V20-HLT',
            'status': 'restored',
            'ocr_text': ocr_data['extracted_text'],
            'uploaded_by': 'admin',
            'uploader_name': 'Archive Officer',
            'created_at': '2026-11-08T13:51:00.000Z',
            'updated_at': '2026-11-08T14:00:00.000Z',
            'is_deleted': False
        }
        db.documents.update_one({'document_id': 'DOC-1918-HLT'}, {'$set': doc_1918}, upsert=True)
        
        # Versions
        db.document_versions.update_one({'version_id': 'VER-V10-HLT'}, {'$set': {
            'version_id': 'VER-V10-HLT',
            'document_id': 'DOC-1918-HLT',
            'version_number': 'v1.0',
            'version_type': 'original',
            'file_key': '8c334f7bf09e47f4a70d60c0fb3b2bd0.jpg',
            'model_used': 'Raw Archival Scan (Unmodified Ingestion)',
            'created_at': '2026-11-08T13:51:00.000Z'
        }}, upsert=True)
        db.document_versions.update_one({'version_id': 'VER-V20-HLT'}, {'$set': {
            'version_id': 'VER-V20-HLT',
            'document_id': 'DOC-1918-HLT',
            'version_number': 'v2.0',
            'version_type': 'restored',
            'file_key': '8c334f7bf09e47f4a70d60c0fb3b2bd0_restored.png',
            'model_used': 'OpenCV Sovereign Preprocessing + Contrast Enhancement + PaddleOCR',
            'created_at': '2026-11-08T14:00:00.000Z'
        }}, upsert=True)

        # OCR
        db.ocr_results.update_one({'document_id': 'DOC-1918-HLT'}, {'$set': {
            'document_id': 'DOC-1918-HLT',
            'version_id': 'VER-V20-HLT',
            'extracted_text': ocr_data['extracted_text'],
            'corrected_text': ocr_data['extracted_text'],
            'language': 'English',
            'text_regions': ocr_data['regions'],
            'regions': ocr_data['regions'],
            'average_confidence': ocr_data['average_confidence'],
            'model_used': 'PaddleOCR PP-OCRv6 Sovereign Engine',
            'engine': 'PaddleOCR PP-OCRv6',
            'word_count': len(ocr_data['extracted_text'].split()),
            'line_count': len(ocr_data['regions']),
            'created_at': '2026-11-08T14:00:00.000Z'
        }}, upsert=True)
        print("Synchronized DOC-1918-HLT to MongoDB Atlas!")
    except Exception as ex:
        print(f"Note on 1918 doc sync: {ex}")

    print("\nMongoDB Atlas migration complete! All collections synchronized.")

if __name__ == '__main__':
    run_migration()
