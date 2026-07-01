import os
from pypdf import PdfReader # Let's see if pypdf or PyPDF2 is installed

pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]

for pdf_file in pdf_files:
    try:
        reader = PdfReader(pdf_file)
        print(f"\n--- PDF: {pdf_file} ({len(reader.pages)} paginas) ---")
        for idx, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if "carralancha" in text.lower() or "v4" in text.lower() or "pozaldez" in text.lower():
                print(f"  Pagina {idx}:")
                for line in text.split("\n"):
                    if any(x in line.lower() for x in ["carralancha", "v4", "pozaldez", "sector", "superficie", "area"]):
                        print(f"    {line.strip()}")
    except Exception as e:
        print(f"Error en {pdf_file}: {e}")
