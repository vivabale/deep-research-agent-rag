from pathlib import Path
import logging
from .loaders.factory import get_loader

logger=logging.getLogger(__name__)

class KnowledgeLoader:
    def __init__(self, directory="knowledge_base"):
        self.directory=Path(directory)

    def load(self):
        docs=[]
        if not self.directory.exists():
            logger.warning("Knowledge base not found: %s", self.directory)
            return docs
        for path in self.directory.rglob("*"):
            if not path.is_file():
                continue
            loader=get_loader(path)
            if loader is None:
                logger.warning("Unsupported file skipped: %s", path.name)
                continue
            try:
                docs.append(loader.load(path))
            except Exception as e:
                logger.warning("Failed loading %s: %s", path.name, e)
        return docs
