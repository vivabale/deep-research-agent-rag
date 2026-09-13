from pathlib import Path
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader

LOADERS={".txt":TextLoader,".md":TextLoader,".pdf":PDFLoader,".docx":DocxLoader}

def get_loader(path):
    cls=LOADERS.get(Path(path).suffix.lower())
    return cls() if cls else None
