# Search Engine From Scratch

### How To Run



### 1. Adding Elias-Gamma Compression

Comparing 5 types of compression
```
Codec                                  Size (KB)     Time (s)
--------------------------------------------------------------
StandardPostings                        1557.372        2.281
VBEPostings                             1024.439        2.543
EliasGammaPostings                      1022.873        3.377
VBEPostingsEliasGammaTF                  999.328        3.100
EliasGammaPostingsVBETF                 1048.614        3.104
```

Conclusion: VBEPostingsEliasGammaTF -> smallest size

### 2. Adding BM25

Comparing TF-IDF with BM25 over 30 queries
```
TF-IDF RBP = 0.5980
BM25   RBP = 0.6317
```

### 3. Adding 3 More Evaluation Metrics: NDCG, DCG, & AP

Evaluation results over 30 queries:
```
TF-IDF RBP  = 0.6052398617530756
TF-IDF DCG  = 5.422493684952236
TF-IDF NDCG = 0.7904113437261435
TF-IDF AP   = 0.518025692430456
BM25   RBP  = 0.6467038364528055
BM25   DCG  = 5.613506000661405
BM25   NDCG = 0.8144864857338686
BM25   AP   = 0.5558783234012998
```