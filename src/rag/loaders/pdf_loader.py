from pathlib import Path
from pypdf import PdfReader
from ..schemas import Document
from .base import BaseLoader

class PDFLoader(BaseLoader):
    def load(self,path):
        p=Path(path)
        reader=PdfReader(path)
        return Document("\n".join((x.extract_text() or "") for x in reader.pages), {"source":p.name,"filename":p.name,"filetype":p.suffix.lower()})
