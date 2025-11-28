from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from .score import analyze_resume
from django.http import JsonResponse
import os

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return extract_pdf_text(file_path)
    
    elif ext == '.docx':
        doc = Document(file_path)
        return "\n" .join(p.text for p in doc.paragraphs)
    
    elif ext == '.txt':
        with open (file_path, 'r', encoding='utf-8') as f:
            return f.read()

    return ""


def upload_resume(request):
    if request.method == "POST":
        resume = request.FILES["resume"]
        fs = FileSystemStorage()
        saved_path = fs.save(resume.name, resume)
        full_path = fs.path(saved_path)
        extracted_text = extract_text(full_path)
        result = analyze_resume(extracted_text)

        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            # Return JSON for AJAX
            return JsonResponse({
                "text": extracted_text,
                "result": result
            })

        return render(request, "result.html", {"text": extracted_text, "result": result})
    return render(request, "upload.html")