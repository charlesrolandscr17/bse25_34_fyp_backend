# import fitz  # PyMuPDF


# def read_pdf(file_path):
#     try:
#         doc = fitz.open(file_path)
#         text = ""
#         for page in doc:
#             text += page.get_text()
#         doc.close()
#         return text
#     except Exception as e:
#         return f"Error reading PDF: {e}"


# if __name__ == "__main__":
#     file_path = "/home/croland/Downloads/Delivery Manager - SAP Platforms (Krishna Kammili) 03132025 _ 9737.pdf"  # Replace with your PDF path
#     content = read_pdf(file_path)
#     print(content)

import requests
import fitz  # PyMuPDF
import io


def download_pdf_content(url: str) -> bytes:
    """Fetches PDF content from a URL and returns it as bytes."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise RuntimeError(f"Failed to download PDF: {e}")


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts and returns text from PDF bytes."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")
    
def get_text(url):
    """Fetches PDF content from a URL and extracts text."""
    pdf_bytes = download_pdf_content(url)
    return extract_text_from_pdf_bytes(pdf_bytes)


if __name__ == "__main__":
    url = "https://tvsgsvnqdahjnvwdnufk.supabase.co/storage/v1/object/public/resume//Delivery%20Manager%20-%20SAP%20Platforms%20(Krishna%20Kammili)%2003132025%20_%209737.pdf"  # Replace with your own
    # pdf_bytes = download_pdf_content(url)
    text = get_text(url)
    print(text)
