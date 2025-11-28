import os
import pdfplumber
import pandas as pd

PDF_FOLDER = "resume_dataset/pdfs"
OUTPUT_CSV = "resume_labels.csv"

def extract_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except:
        text = ""
    return text

def main():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    data = []

    print(f"Found {len(pdf_files)} PDF resumes. Extracting text...\n")

    for file in pdf_files:
        full_path = os.path.join(PDF_FOLDER, file)
        text = extract_text(full_path)

        data.append({
            "filename": file,
            "text": text,
            "label": ""   # you will manually mark good / bad later
        })

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDONE! CSV saved as: {OUTPUT_CSV}")
    print("Please open it and fill the label column with: good or bad")

if __name__ == "__main__":
    main()
