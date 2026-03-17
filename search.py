from bsbi import BSBIIndex
from compression import VBEPostingsEliasGammaTF

# indexing has already been performed
# BSBIIndex is only an abstraction for that index
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostingsEliasGammaTF, \
                          output_dir = 'index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
           
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        print(f"{doc:30} {score:>.3f}")
    print()