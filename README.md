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