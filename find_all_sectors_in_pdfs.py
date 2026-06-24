import os
import pypdf
import re

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

# Find all PDF files in workspace
pdf_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".pdf")]

print("--- SCANNING FOR ALL SECTORS ---")
for path in pdf_files:
    filename = os.path.basename(path)
    try:
        reader = pypdf.PdfReader(path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Search for "sector" or any number of sector
            matches = list(re.finditer(r'sector\s*\d+', text, re.IGNORECASE))
            if matches:
                print(f"\nFile: {filename}, Page: {idx+1}")
                for m in matches:
                    start = max(0, m.start() - 100)
                    end = min(len(text), m.end() + 100)
                    context = text[start:end].replace('\n', ' ')
                    print(f"  Match: ... {context} ...")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
