import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


try:
    import faiss
except ImportError as exc:
    raise SystemExit(
        "faiss is required. Install with: pip install faiss-cpu"
    ) from exc


TOKEN_PATTERN = re.compile(r"\w+")


def simple_tokenizer(text):
    return TOKEN_PATTERN.findall(text.lower())


def iter_document_paths(collection_dir):
    base = Path(collection_dir)
    for path in sorted(base.rglob("*.txt"), key=lambda p: (str(p.parent), p.name)):
        yield path


def build_sparse_tfidf(doc_paths, min_df=2, max_df=0.9):
    # input='filename' lets scikit-learn stream file reads internally.
    vectorizer = TfidfVectorizer(
        input="filename",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
    )
    X = vectorizer.fit_transform([str(p) for p in doc_paths])
    # Keep filename streaming for build, but store vectorizer in content mode
    # so query text is interpreted as plain text (not a file path).
    vectorizer.input = "content"
    return vectorizer, X


def build_faiss_index(vectors, index_type="ivf", nlist=256, m=32, ef_construction=200):
    dim = vectors.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(vectors)
        return index

    if index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        n_docs = vectors.shape[0]
        effective_nlist = max(1, min(nlist, int(np.sqrt(max(n_docs, 1)) * 8)))
        index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT)

        # Train IVF on a sample if document count is large.
        train_size = min(n_docs, max(10000, effective_nlist * 40))
        if n_docs > train_size:
            rng = np.random.default_rng(42)
            train_ids = rng.choice(n_docs, size=train_size, replace=False)
            train_data = vectors[train_ids]
        else:
            train_data = vectors

        index.train(train_data)
        index.nprobe = min(16, effective_nlist)
        index.add(vectors)
        return index

    raise ValueError(f"Unsupported index_type: {index_type}")


def build_lsi(
    collection_dir,
    output_dir,
    n_components=256,
    min_df=2,
    max_df=0.9,
    index_type="ivf",
    nlist=256,
    hnsw_m=32,
    ef_construction=200,
):
    doc_paths = list(iter_document_paths(collection_dir))
    if not doc_paths:
        raise ValueError(f"No .txt files found in {collection_dir}")

    vectorizer, X = build_sparse_tfidf(doc_paths, min_df=min_df, max_df=max_df)

    # Randomized TruncatedSVD is efficient for large sparse matrices.
    max_rank = min(X.shape[0] - 1, X.shape[1] - 1)
    if max_rank <= 1:
        raise ValueError("Not enough data to perform SVD")

    n_components = min(n_components, max_rank)
    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        n_iter=7,
        random_state=42,
    )
    doc_lsi = svd.fit_transform(X).astype(np.float32)
    doc_lsi = normalize(doc_lsi, norm="l2", copy=False)

    index = build_faiss_index(
        doc_lsi,
        index_type=index_type,
        nlist=nlist,
        m=hnsw_m,
        ef_construction=ef_construction,
    )

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, str(Path(output_dir) / "docs.faiss"))

    meta = {
        "doc_paths": [str(p).replace("\\", "/") for p in doc_paths],
        "n_components": n_components,
        "explained_variance_ratio": float(np.sum(svd.explained_variance_ratio_)),
        "index_type": index_type,
        "n_docs": len(doc_paths),
        "vocab_size": int(X.shape[1]),
    }

    with open(Path(output_dir) / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(Path(output_dir) / "svd.pkl", "wb") as f:
        pickle.dump(svd, f)

    with open(Path(output_dir) / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    return meta


def load_lsi(output_dir):
    out = Path(output_dir)
    index = faiss.read_index(str(out / "docs.faiss"))

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Old indexes may pickle tokenizer as __main__.simple_tokenizer
            # when built via `python lsi_faiss.py ...`.
            if module == "__main__" and name == "simple_tokenizer":
                return simple_tokenizer
            return super().find_class(module, name)

    with open(out / "vectorizer.pkl", "rb") as f:
        try:
            vectorizer = pickle.load(f)
        except (AttributeError, ModuleNotFoundError):
            f.seek(0)
            vectorizer = _CompatUnpickler(f).load()

    with open(out / "svd.pkl", "rb") as f:
        svd = pickle.load(f)

    with open(out / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    return index, vectorizer, svd, meta


def query_lsi(output_dir, text, topk=10):
    index, vectorizer, svd, meta = load_lsi(output_dir)
    # Backward compatibility for indexes built before the input-mode fix.
    if getattr(vectorizer, "input", None) == "filename":
        vectorizer.input = "content"
    q_tfidf = vectorizer.transform([text])
    q_lsi = svd.transform(q_tfidf).astype(np.float32)
    q_lsi = normalize(q_lsi, norm="l2", copy=False)

    scores, doc_ids = index.search(q_lsi, topk)
    hits = []
    for score, doc_id in zip(scores[0], doc_ids[0]):
        if doc_id < 0:
            continue
        hits.append((float(score), meta["doc_paths"][int(doc_id)]))
    return hits


def build_parser():
    parser = argparse.ArgumentParser(
        description="LSI retrieval with sparse TF-IDF + randomized SVD + FAISS"
    )
    sub = parser.add_subparsers(dest="command", required=False)
    parser.set_defaults(command="build")

    p_build = sub.add_parser("build", help="Build LSI + FAISS document index")
    p_build.add_argument("--collection", default="collection", help="Collection root directory")
    p_build.add_argument("--output-dir", default="lsi_index", help="Output directory for LSI artifacts")
    p_build.add_argument("--n-components", type=int, default=256, help="Latent dimensions for SVD")
    p_build.add_argument("--min-df", type=int, default=2, help="Minimum document frequency")
    p_build.add_argument("--max-df", type=float, default=0.9, help="Maximum document frequency ratio")
    p_build.add_argument(
        "--index-type",
        choices=["flat", "ivf", "hnsw"],
        default="ivf",
        help="FAISS index type",
    )
    p_build.add_argument("--nlist", type=int, default=256, help="IVF cluster count (upper bound)")
    p_build.add_argument("--hnsw-m", type=int, default=32, help="HNSW neighbor count")
    p_build.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW construction effort",
    )

    p_query = sub.add_parser("query", help="Search query using built LSI+FAISS index")
    p_query.add_argument("--output-dir", default="lsi_index", help="Directory holding LSI artifacts")
    p_query.add_argument("--text", required=True, help="Query text")
    p_query.add_argument("--topk", type=int, default=10, help="Top-k results")

    return parser


def main():
    args = build_parser().parse_args()

    # Provide defaults for build subcommand when not explicitly invoked
    if args.command == "build":
        if not hasattr(args, "collection"):
            args.collection = "collection"
        if not hasattr(args, "output_dir"):
            args.output_dir = "lsi_index"
        if not hasattr(args, "n_components"):
            args.n_components = 256
        if not hasattr(args, "min_df"):
            args.min_df = 2
        if not hasattr(args, "max_df"):
            args.max_df = 0.9
        if not hasattr(args, "index_type"):
            args.index_type = "ivf"
        if not hasattr(args, "nlist"):
            args.nlist = 256
        if not hasattr(args, "hnsw_m"):
            args.hnsw_m = 32
        if not hasattr(args, "ef_construction"):
            args.ef_construction = 200

        meta = build_lsi(
            collection_dir=args.collection,
            output_dir=args.output_dir,
            n_components=args.n_components,
            min_df=args.min_df,
            max_df=args.max_df,
            index_type=args.index_type,
            nlist=args.nlist,
            hnsw_m=args.hnsw_m,
            ef_construction=args.ef_construction,
        )
        print("LSI index built")
        print(f"- docs: {meta['n_docs']}")
        print(f"- vocab: {meta['vocab_size']}")
        print(f"- latent dims: {meta['n_components']}")
        print(f"- explained variance ratio: {meta['explained_variance_ratio']:.4f}")
        print(f"- faiss index: {meta['index_type']}")
        return

    if args.command == "query":
        hits = query_lsi(args.output_dir, args.text, topk=args.topk)
        print(f"Query: {args.text}")
        for score, path in hits:
            print(f"{path:40} {score:>.4f}")


if __name__ == "__main__":
    main()
