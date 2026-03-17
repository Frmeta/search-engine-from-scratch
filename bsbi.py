import os
import pickle
import contextlib
import heapq
import time
import math
import shutil
import argparse

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import (
    StandardPostings,
    VBEPostings,
    EliasGammaPostings,
    VBEPostingsEliasGammaTF,
    EliasGammaPostingsVBETF,
)
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Maps terms to termIDs
    doc_id_map(IdMap): Maps relative document paths (e.g.,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path to data
    output_dir(str): Path to output index files
    postings_encoding: See compression.py; candidates include StandardPostings,
                    VBEPostings, etc.
    index_name(str): Name of the file containing the inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Stores file names for all intermediate inverted indices
        self.intermediate_indices = []

    def save(self):
        """Save doc_id_map and term_id_map to output directory via pickle."""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Load doc_id_map and term_id_map from output directory."""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Parse text files into a sequence of
        <termID, docID> pairs.

        Use available tools for English stemming.

        Do not forget to remove stopwords.

        For sentence segmentation and tokenization, you may use regex
        or other machine-learning-based tools.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to the directory containing text files for one block.

            Note that one folder in collection is treated as one block.
            The block concept in this assignment differs from the block concept
            in operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Returns all <termID, docID> pairs from a block (a subdirectory
            in the collection folder).

        Must use self.term_id_map and self.doc_id_map to obtain termIDs
        and docIDs. These two variables must persist across all calls to
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Invert td_pairs (a list of <termID, docID> pairs) and
        write them to index. This applies the BSBI concept where
        one large dictionary is maintained per block.
        The storage strategy uses SPIMI-like hashtable structures
        (implemented as Python dictionaries).

        ASSUMPTION: td_pairs fit in memory.

        In Programming Assignment 1, we only added terms and
        sorted Doc ID lists. In Programming Assignment 2,
        we also need to add TF lists.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index on disk (file) associated with a block
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Merge all intermediate inverted indices into
        a single index.

        This is the EXTERNAL MERGE SORT step.

        Use function sorted_merge_posts_and_tfs(..) in util module.

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, each
            representing an iterable intermediate inverted index
            for a block.

        merged_index: InvertedIndexWriter
            InvertedIndexWriter instance that stores the merge result
            from all intermediate InvertedIndexWriter objects.
        """
        # The following code assumes there is at least one term.
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Perform ranked retrieval with TaaT (Term-at-a-Time).
        Returns top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       if tf(t, D) > 0
            = 0                        otherwise

        w(t, Q) = IDF = log (N / df(t))

        Score = for each query term, accumulate w(t, Q) * w(t, D).
            (no document-length normalization required)

        notes:
            1. DF(t) is available in merged_index.postings_dict
            2. TF(t, D) is available in tf_list
            3. N can be obtained from merged_index.doc_length, i.e., len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens separated by spaces

            example: Query "universitas indonesia depok" contains
            three terms: universitas, indonesia, and depok

        Result
        ------
        List[(int, str)]
            List of tuples: first element is similarity score, and
            second is document name.
            Top-K documents sorted by decreasing score.

        Do not raise errors/exceptions for terms that are not in the collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def index(self):
        """
        Base indexing code
        Main section for indexing using BSBI (blocked-sort
        based indexing)

        This method scans all data in collection, calls parse_block
        to parse documents, and calls invert_write to perform inversion
        for each block and write to a new index.
        """
        # Loop over each sub-directory in collection folder (each block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSBI indexing runner")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Compare all postings encodings (size in KB and indexing time)",
    )
    args = parser.parse_args()

    def clean_output_dir(output_dir):
        """Remove old index files so benchmark comparison is fair."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            return

        for filename in os.listdir(output_dir):
            if filename == '.gitkeep':
                continue
            path = os.path.join(output_dir, filename)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

    def non_intermediate_index_size_kb(output_dir):
        """Compute total file size (KB) in output_dir excluding intermediate_index*."""
        total_bytes = 0
        for filename in os.listdir(output_dir):
            if filename.startswith('intermediate_index'):
                continue
            path = os.path.join(output_dir, filename)
            if os.path.isfile(path):
                total_bytes += os.path.getsize(path)
        return total_bytes / 1024.0

    if args.debug:
        codecs = [
            StandardPostings,
            VBEPostings,
            EliasGammaPostings,
            VBEPostingsEliasGammaTF,
            EliasGammaPostingsVBETF,
        ]

        results = []
        for codec in codecs:
            clean_output_dir('index')
            start = time.perf_counter()
            BSBIIndex(
                data_dir='collection',
                postings_encoding=codec,
                output_dir='index'
            ).index()
            elapsed = time.perf_counter() - start
            size_kb = non_intermediate_index_size_kb('index')
            results.append((codec.__name__, size_kb, elapsed))

        print("\npostings_encoding comparison (non-intermediate size & indexing time)")
        print(f"{'Codec':35} {'Size (KB)':>12} {'Time (s)':>12}")
        print("-" * 62)
        for name, size_kb, elapsed in results:
            print(f"{name:35} {size_kb:12.3f} {elapsed:12.3f}")
    else:
        BSBI_instance = BSBIIndex(
            data_dir='collection',
            postings_encoding=VBEPostingsEliasGammaTF,
            output_dir='index'
        )
        BSBI_instance.index()  # start indexing
