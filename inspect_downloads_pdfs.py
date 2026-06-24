import re
import pypdf
import os

pdf_files = [
    r"C:\Users\fdemiguel\Downloads\Riego.pdf",
    r"C:\Users\fdemiguel\Downloads\Riego_CR.pdf"
]

keywords = ["sector 2", "sector2", "caudal", "superficie", "area", "v2", "m2", "ha", "litro", "metro", "flow", "canduela"]

print("--- SCANNING DOWNLOADS PDFs ---")
for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        print(f"\nScanning: {pdf_path}")
        try:
            reader = pypdf.PdfReader(pdf_path)
            print(f"Total pages: {len(reader.pages)}")
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                # Search for keywords
                found_kws = [kw for kw in keywords if kw in text.lower()]
                if found_kws:
                    print(f"  Page {idx+1} matches: {found_kws}")
                    # Print lines containing matches
                    lines = text.split('\n')
                    for line in lines:
                        if any(kw in line.lower() for kw in keywords):
                            print("    ", line.strip()[:150])
        except Exception as e:
            print("  Error:", e)
    else:
        print(f"File not found: {pdf_path}")
