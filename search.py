import argparse
from bsbi import BSBIIndex, SPIMIIndex
from compression import VBEPostingsEliasGammaTF

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

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
           
for query in queries:
    print("Query  : ", query)
    print("TF-IDF Results:")
    for (score, doc) in index_instance.retrieve_tfidf(query, k = 10, use_patricia = args.patricia):
         print(f"{doc:30} {score:>.3f}")
    print("BM25 Results:")
    for (score, doc) in index_instance.retrieve_bm25(query, k = 10, use_patricia = args.patricia):
        print(f"{doc:30} {score:>.3f}")
    print()