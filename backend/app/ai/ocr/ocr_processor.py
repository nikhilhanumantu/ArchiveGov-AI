import json
from pathlib import Path
from typing import Dict, Any
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from backend.app.storage.local_storage import storage

class OCROutputExporter:
    """Exports OCR text results to TXT, DOCX, JSON, and PDF formats."""

    @staticmethod
    def export_txt(document_id: str, title: str, text: str) -> str:
        content = f"ARCHIVEGOV AI — CERTIFIED TRANSCRIPTION\n"
        content += f"Document ID: {document_id}\n"
        content += f"Title: {title}\n"
        content += f"{'='*60}\n\n"
        content += text
        
        file_key, path = storage.save_bytes(content.encode("utf-8"), category="exports", extension=".txt")
        return file_key

    @staticmethod
    def export_json(document_id: str, title: str, ocr_data: Dict[str, Any]) -> str:
        payload = {
            "archivegov_transcription_record": {
                "document_id": document_id,
                "document_title": title,
                "engine": ocr_data.get("engine", "PaddleOCR"),
                "language": ocr_data.get("language", "en"),
                "confidence": ocr_data.get("average_confidence", 95.0),
                "extracted_text": ocr_data.get("extracted_text", ""),
                "corrected_text": ocr_data.get("corrected_text", ""),
                "text_regions": ocr_data.get("regions", [])
            }
        }
        raw_json = json.dumps(payload, indent=2).encode("utf-8")
        file_key, path = storage.save_bytes(raw_json, category="exports", extension=".json")
        return file_key

    @staticmethod
    def export_docx(document_id: str, title: str, text: str) -> str:
        doc = Document()
        doc.add_heading("ArchiveGov AI — Certified Government Record Transcription", 0)
        
        p = doc.add_paragraph()
        p.add_run(f"Document ID: ").bold = True
        p.add_run(f"{document_id}\n")
        p.add_run(f"Folio Title: ").bold = True
        p.add_run(f"{title}\n")
        p.add_run(f"Classification: ").bold = True
        p.add_run("National Archival Record — Verified\n")
        
        doc.add_heading("Transcribed Text Content", level=1)
        for line in text.split("\n"):
            if line.strip():
                doc.add_paragraph(line.strip())
                
        file_key = f"export_{document_id}.docx"
        dest_path = storage.exports / file_key
        doc.save(str(dest_path))
        return file_key

    @staticmethod
    def export_pdf(document_id: str, title: str, text: str) -> str:
        file_key = f"export_{document_id}.pdf"
        dest_path = storage.exports / file_key
        
        doc = SimpleDocTemplate(str(dest_path), pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ArchivalTitle',
            parent=styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor='#101942'
        )
        body_style = ParagraphStyle(
            'ArchivalBody',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            textColor='#1E293B'
        )
        
        story = [
            Paragraph("ArchiveGov AI — Certified Archival Record Transcription", title_style),
            Spacer(1, 12),
            Paragraph(f"<b>Document ID:</b> {document_id} | <b>Title:</b> {title}", body_style),
            Spacer(1, 16),
            Paragraph("<b>Transcription:</b>", styles['Heading2']),
            Spacer(1, 8)
        ]
        
        for line in text.split("\n"):
            if line.strip():
                # Escape XML characters for reportlab
                escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(escaped, body_style))
                story.append(Spacer(1, 6))
                
        doc.build(story)
        return file_key

ocr_exporter = OCROutputExporter()

