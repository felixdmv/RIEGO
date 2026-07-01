from pypdf import PdfReader
import os

pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
for f in pdf_files:
    reader = PdfReader(f)
    txt = reader.pages[0].extract_text()
    print(f"{f}: {len(txt)} chars extracted")
    if len(txt) > 0:
        print("Sample:", txt[:100].replace("\n", " "))
