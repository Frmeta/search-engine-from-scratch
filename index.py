import pickle
import os

class InvertedIndex:
    """
    Class that implements efficient scanning/reading of an inverted index
    stored in a file, and also provides a mechanism to write an
    inverted index to storage during indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list)

          postings_dict is the Dictionary component of the inverted index.
          It is assumed this structure can be fully loaded in memory.

          As the name implies, this Dictionary is implemented as a Python dict
          mapping a term ID (integer) to a 4-tuple:
              1. start_position_in_index_file: byte offset where the associated
                  postings are stored in the index file; can be reached with seek.
              2. number_of_postings_in_list: number of docIDs in the postings list
                  (Document Frequency)
              3. length_in_bytes_of_postings_list: byte length of the postings list
              4. length_in_bytes_of_tf_list: byte length of the associated TF list

    terms: List[int]
        List of term IDs used to preserve insertion order in the inverted index.

    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Name used for index storage files
        postings_encoding : See compression.py, candidates include StandardPostings,
                GapBasedPostings, etc.
        directory (str): Directory where index files are stored
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # Keep track of insertion order of terms
        self.doc_length = {}    # key: doc ID (int), value: document length (number of tokens)
                    # Useful for score normalization by document length
                    # when computing TF-IDF or BM25 scores
        self.total_doc_length = 0
        self.avg_doc_length = 0.0

    def __enter__(self):
        """
        Load all metadata when entering the context manager.
        Metadata:
            1. Dictionary ---> postings_dict
            2. iterator over list of term insertion order in the index
                during construction ---> term_iter
            3. doc_length, a Python dictionary with key = doc_id and
                value = number of tokens in that document (document length).
                Useful for length normalization in TF-IDF/BM25 scoring and
                for obtaining N in IDF computation, where N is number of docs.

        Metadata is persisted using the "pickle" library.

        See Python __enter__(..) and context manager docs:

        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Open index file
        self.index_file = open(self.index_file_path, 'rb+')

        # Load postings dict and terms iterator from metadata file
        with open(self.metadata_file_path, 'rb') as f:
            metadata = pickle.load(f)
            if len(metadata) == 3:
                self.postings_dict, self.terms, self.doc_length = metadata
                self.total_doc_length = sum(self.doc_length.values())
                self.avg_doc_length = self.total_doc_length / len(self.doc_length) if self.doc_length else 0.0
            elif len(metadata) == 4:
                self.postings_dict, self.terms, self.doc_length, self.avg_doc_length = metadata
                self.total_doc_length = sum(self.doc_length.values())
            else:
                self.postings_dict, self.terms, self.doc_length, self.total_doc_length, self.avg_doc_length = metadata
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Close index_file and save postings_dict and terms on context exit."""
        # Close index file
        self.index_file.close()

        if self.total_doc_length == 0 and self.doc_length:
            self.total_doc_length = sum(self.doc_length.values())
        self.avg_doc_length = self.total_doc_length / len(self.doc_length) if self.doc_length else 0.0

        # Save metadata (postings dict and terms) to metadata file using pickle
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([
                self.postings_dict,
                self.terms,
                self.doc_length,
                self.total_doc_length,
                self.avg_doc_length,
            ], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class that implements efficient scanning/reading
    of an inverted index stored in a file.
    """
    def __iter__(self):
        return self

    def reset(self):
        """
        Reset file pointer and term iterator to the beginning.
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__() # reset term iterator

    def __next__(self):
        """
        InvertedIndexReader is iterable (has an iterator).
        See:
        https://stackoverflow.com/questions/19151/how-to-build-a-basic-iterator

        When an instance of this class is used in a loop, __next__(...)
        returns the next (term, postings_list, tf_list) tuple
        from the inverted index.

        IMPORTANT: this method must return only a small part of a large
        index file so it fits in memory. Do not load the entire index.
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Return postings list (list of docIDs) and corresponding
        term-frequency list for a term as tuple (postings_list, tf_list).

        IMPORTANT: this method must not iterate over the entire index.
        It should directly jump to the relevant byte offset in the index file
        where postings and TF list for the term are stored.
        """
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[term]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Class that implements efficient writing
    of an inverted index into a file.
    """
    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
          Append a term, its postings_list, and associated TF list
          to the end of the index file.

          This method performs 4 steps:
        1. Encode postings_list using self.postings_encoding (method encode),
        2. Encode tf_list using self.postings_encoding (method encode_tf),
          3. Store metadata in self.terms, self.postings_dict, and self.doc_length.
              Recall that self.postings_dict maps termID to a 4-tuple:
                 - start_position_in_index_file
                 - number_of_postings_in_list
                 - length_in_bytes_of_postings_list
                 - length_in_bytes_of_tf_list
          4. Append encoded postings bytestream and encoded TF bytestream
              to the end of the index file on disk.

          Do not forget to update self.terms and self.doc_length.

        SEARCH ON YOUR FAVORITE SEARCH ENGINE:
                - You may want to read about Python I/O
          https://docs.python.org/3/tutorial/inputoutput.html
                    This link also explains how to append data to the end of a file.
                - Useful file object methods include seek(...) and tell().

        Parameters
        ----------
        term:
            Term or termID, a unique identifier for a term
        postings_list: List[Int]
            List of docIDs where the term appears
        tf_list: List[Int]
            List of term frequencies
        """
        self.terms.append(term) # update self.terms

        # update self.doc_length
        for i in range(len(postings_list)):
            doc_id, freq = postings_list[i], tf_list[i]
            if doc_id not in self.doc_length:
                self.doc_length[doc_id] = 0
            self.doc_length[doc_id] += freq
            self.total_doc_length += freq

        self.index_file.seek(0, os.SEEK_END)
        curr_position_in_byte = self.index_file.tell()
        compressed_postings = self.postings_encoding.encode(postings_list)
        compressed_tf_list = self.postings_encoding.encode_tf(tf_list)
        self.index_file.write(compressed_postings)
        self.index_file.write(compressed_tf_list)
        self.postings_dict[term] = (curr_position_in_byte, len(postings_list), \
                                    len(compressed_postings), len(compressed_tf_list))


if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1,2], "terms are incorrect"
        assert index.doc_length == {2:2, 3:38, 4:25, 5:56, 8:3, 10:30}, "doc_length is incorrect"
        assert index.postings_dict == {1: (0, \
                                           5, \
                                           len(VBEPostings.encode([2,3,4,8,10])), \
                                           len(VBEPostings.encode_tf([2,4,2,3,30]))),
                                       2: (len(VBEPostings.encode([2,3,4,8,10])) + len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           3, \
                                           len(VBEPostings.encode([3,4,5])), \
                                           len(VBEPostings.encode_tf([34,23,56])))}, "postings dictionary is incorrect"
        
        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3,4,5])))) == [3,4,5], "there is an error"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34,23,56])))) == [34,23,56], "there is an error"
