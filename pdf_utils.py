from pdf2image import convert_from_bytes

def extract_pdf_pages(pdf_bytes, dpi=200):
    return convert_from_bytes(pdf_bytes, dpi=dpi)
