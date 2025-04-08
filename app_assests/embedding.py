import fitz
from docx import Document as DocxDocument
import pandas as pd
import os

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_from_excel(file_path)
    else:
        raise ValueError("Unsupported file type")

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_excel(file_path: str) -> str:
    dfs = pd.read_excel(file_path, sheet_name=None)
    combined_text = ""
    for name, df in dfs.items():
        combined_text += f"\nSheet: {name}\n"
        combined_text += df.astype(str).to_string(index=False)
    return combined_text

def get_embedding(text: str):
    return model.encode(text).tolist()
