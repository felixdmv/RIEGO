import os
import re

base_path = r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\RIEGO"

# Find all PDF files in workspace
pdf_files = []
for f in os.listdir(base_path):
    if f.endswith(".pdf"):
        pdf_files.append(os.path.join(base_path, f))

print("Found PDFs:", [os.path.basename(p) for p in pdf_files])

# Try to extract text using standard libraries or simple binary scanning for ASCII text if no libraries are available
try:
    import pypdf
    print("Using pypdf")
    def get_pdf_text(path):
        reader = pypdf.PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
except ImportError:
    try:
        import PyPDF2
        print("Using PyPDF2")
        def get_pdf_text(path):
            reader = PyPDF2.PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except ImportError:
        print("No PDF library installed, using fallback binary scanner")
        def get_pdf_text(path):
            with open(path, "rb") as f:
                content = f.read()
            # Clean up printable ascii strings
            ascii_strings = re.findall(b'[\x20-\x7E\x0A\x0D]{4,}', content)
            return " ".join(s.decode('ascii', errors='ignore') for s in ascii_strings)

keywords = ["sector 2", "caudal", "superficie", "canduela", "valvula 2", "v2", "m3/h", "hectarea", "area"]

for path in pdf_files:
    print(f"\n--- Scanning {os.path.basename(path)} ---")
    try:
        text = get_pdf_text(path)
        # Search for coords, flow, area
        for kw in keywords:
            matches = re.findall(r'.{0,100}' + re.escape(kw) + r'.{0,100}', text, re.IGNORECASE)
            if matches:
                print(f"Keyword '{kw}' found:")
                for m in matches[:5]:
                    print("  ...", m.strip().replace('\n', ' '), "...")
    except Exception as e:
        print(f"Error scanning {os.path.basename(path)}: {e}")
