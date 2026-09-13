import logging
from .loader import KnowledgeLoader
from .splitters.selector import get_splitter
from .embeddings import LocalEmbedding
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def build():
    docs=KnowledgeLoader().load()
    chunks=[]
    for doc in docs:
        splitter=get_splitter(doc.metadata.get("filetype",""),doc.content)
        chunks.extend(splitter.split(doc))
    logger.info("documents=%s chunks=%s",len(docs),len(chunks))
    vectors=LocalEmbedding().embed_documents([c.content for c in chunks])
    store=VectorStore(); store.add(chunks,vectors); store.persist()
    return len(chunks)

if __name__=="__main__": print(build())
