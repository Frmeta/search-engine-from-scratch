import argparse
from bsbi import BSBIIndex, SPIMIIndex
from compression import VBEPostingsEliasGammaTF
from lsi_faiss import query_lsi
from adaptive_retrieval import AdaptiveRetriever

parser = argparse.ArgumentParser(description="Search with BSBI or SPIMI index")
parser.add_argument(
    "--spimi",
    action="store_true",
    help="Use SPIMI indexing instead of BSBI",
)
parser.add_argument(
    "--patricia",
    action="store_true",
    help="Use Patricia Tree for query term lookup",
)
parser.add_argument(
    "--adaptive-retrieval",
    action="store_true",
    default=True,
    help="Show adaptive retrieval results (default: True)",
)
parser.add_argument(
    "--no-adaptive-retrieval",
    dest="adaptive_retrieval",
    action="store_false",
    help="Disable adaptive retrieval results",
)
parser.add_argument(
    "--adaptive-index-dir",
    default="pt_index",
    help="Directory for adaptive retrieval index artifacts",
)
parser.add_argument(
    "--lsi-faiss",
    action="store_true",
    default=True,
    help="Show LSI+FAISS retrieval results (default: True)",
)
parser.add_argument(
    "--no-lsi-faiss",
    dest="lsi_faiss",
    action="store_false",
    help="Disable LSI+FAISS retrieval results",
)
parser.add_argument(
    "--lsi-output-dir",
    default="lsi_index",
    help="Directory for LSI+FAISS artifacts",
)
parser.add_argument(
    "--topk",
    type=int,
    default=10,
    help="Top-k retrieval results",
)
args = parser.parse_args()

# Indexing has already been performed
# BSBIIndex/SPIMIIndex is an abstraction for that index
if args.spimi:
    index_instance = SPIMIIndex(
        data_dir='collection',
        postings_encoding=VBEPostingsEliasGammaTF,
        output_dir='index'
    )
else:
    index_instance = BSBIIndex(
        data_dir='collection',
        postings_encoding=VBEPostingsEliasGammaTF,
        output_dir='index'
    )

adaptive_retriever = None
if args.adaptive_retrieval:
    try:
        adaptive_retriever = AdaptiveRetriever(
            collection_dir='collection',
            index_dir=args.adaptive_index_dir,
        )
    except RuntimeError as e:
        print(f"Warning: {e}", flush=True)
        adaptive_retriever = None

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
           
for query in queries:
    print("Query  : ", query)
    print("TF-IDF Results:")
    for (score, doc) in index_instance.retrieve_tfidf(query, k = args.topk, use_patricia = args.patricia):
         print(f"{doc:30} {score:>.3f}")
    print("BM25 Results:")
    for (score, doc) in index_instance.retrieve_bm25(query, k = args.topk, use_patricia = args.patricia):
        print(f"{doc:30} {score:>.3f}")
    if adaptive_retriever is not None:
        print("Adaptive Retrieval Results:")
        for (score, doc) in adaptive_retriever.retrieve(query, k=args.topk):
            print(f"{doc:30} {score:>.3f}")
    if args.lsi_faiss:
        print("LSI+FAISS Results:")
        try:
            for (score, doc) in query_lsi(args.lsi_output_dir, query, topk=args.topk):
                print(f"{doc:30} {score:>.3f}")
        except Exception as e:
            print(f"Warning: LSI+FAISS query failed: {e}", flush=True)
    print()