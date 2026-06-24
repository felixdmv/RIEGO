import re
import pypdf

pdf_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO - copia\Informe_Final_2024-08-10.pdf"
print("Scanning:", pdf_path)

try:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        text += page_text
        # Search on this page specifically
        if "sector" in page_text.lower() or "caudal" in page_text.lower() or "superficie" in page_text.lower():
            print(f"Match on page {idx+1}:")
            for line in page_text.split('\n'):
                if any(k in line.lower() for k in ["sector", "caudal", "superficie", "canduela", "v2", "m2", "ha", "litro", "metro", "volumen"]):
                    print("  ", line.strip()[:150])
except Exception as e:
    print("Error:", e)
