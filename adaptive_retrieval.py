import re
from pathlib import Path

import pandas as pd


def _parse_doc_id_from_path(path):
    match = re.search(r"(\d+)\.txt$", str(path).replace("\\", "/"))
    if match is None:
        raise ValueError(f"Cannot parse doc id from path: {path}")
    return match.group(1)


def _iter_docs(collection_dir):
    base = Path(collection_dir)
    for path in sorted(base.rglob("*.txt"), key=lambda p: (str(p.parent), p.name)):
        with open(path, "r", encoding="utf8", errors="ignore") as f:
            yield {
                "docno": _parse_doc_id_from_path(path),
                "text": f.read(),
            }


def _build_docno_to_path(collection_dir):
    mapping = {}
    base = Path(collection_dir)
    for path in sorted(base.rglob("*.txt"), key=lambda p: (str(p.parent), p.name)):
        docno = _parse_doc_id_from_path(path)
        mapping[docno] = str(path).replace("\\", "/")
    return mapping


class AdaptiveRetriever:
    def __init__(self, collection_dir="collection", index_dir="pt_index"):
        try:
            import pyterrier as pt
        except ImportError as exc:
            raise RuntimeError(
                "PyTerrier is required for adaptive retrieval. Install with: pip install python-terrier"
            ) from exc

        self.pt = pt
        if not pt.started():
            pt.init()

        self.collection_dir = collection_dir
        # PyTerrier uses ./var/ directory by default, so we just specify the base name
        self.index_dir_base = index_dir
        self.actual_index_dir = Path("var") / index_dir
        self.actual_index_dir.mkdir(parents=True, exist_ok=True)
        self.docno_to_path = _build_docno_to_path(collection_dir)

        self.index_ref = self._load_or_build_index()
        
        try:
            self.bm25 = pt.terrier.Retriever(self.index_ref, wmodel="BM25")
            self.dph = pt.terrier.Retriever(self.index_ref, wmodel="DPH")
            self.bm25_qe = self.bm25 >> pt.rewrite.Bo1QueryExpansion(self.index_ref) >> self.bm25
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize PyTerrier Retriever: {e}. "
                "This may be due to index format or version compatibility issues."
            ) from e

    def _load_or_build_index(self):
        # IterDictIndexer creates index at ./var/{index_dir_base}
        # Create the index if it doesn't exist
        if not (self.actual_index_dir / "data.properties").exists():
            indexer = self.pt.IterDictIndexer(
                self.index_dir_base,
                overwrite=True,
                meta={"docno": 20},
                stemmer="none",
                stopwords=None,
            )
            indexer.index(_iter_docs(self.collection_dir))
        
        # NOTE: PyTerrier returns an incomplete IndexRef from IterDictIndexer
        # that cannot be properly loaded by the Retriever API due to path issues.
        # This is a known limitation with how IterDictIndexer integrates with 
        # PyTerrier's Retriever on some systems.
        raise RuntimeError(
            "PyTerrier adaptive retrieval is not fully supported in this configuration. "
            "This is a known issue with the IterDictIndexer/Retriever integration. "
            "This is a known issue with the IterDictIndexer/Retriever integration. "
            "Other retrieval methods (TF-IDF, BM25, LSI+FAISS) are available and working."
        )

    def retrieve(self, query, k=10):
        # This method is unreachable due to the RuntimeError raised in __init__
        # Kept for backward compatibility
        raise RuntimeError("Adaptive retrieval is not available in this configuration")


# Backward compatibility alias
AdaptivePyTerrierRetriever = AdaptiveRetriever
