import array

class StandardPostings:
    """ 
    Class with static methods to convert a postings list
    from a list of integers into a sequence of bytes.
    It uses Python's array library.

    ASSUMPTION: postings_list for a term fits in memory.

    See:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a byte stream.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            Bytearray representing integer order in postings_list
        """
        # For standard encoding, use L for unsigned long since docIDs
        # are non-negative. Assume maximum docID fits in 4-byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list from a byte stream.

        Parameters
        ----------
        encoded_postings_list: bytes
            Bytearray representing encoded postings list produced
            by static method encode above.

        Returns
        -------
        List[int]
            List of docIDs decoded from encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode term-frequency list into a byte stream.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            Bytearray representing raw TF values of term occurrences
            for each document in the postings list
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode term-frequency list from a byte stream.

        Parameters
        ----------
        encoded_tf_list: bytes
            Bytearray representing encoded term frequencies list produced
            by static method encode_tf above.

        Returns
        -------
        List[int]
            List of term frequencies decoded from encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Different from StandardPostings, where for a postings list,
    the data written to disk is the original integer sequence.

    In VBEPostings, stored values are gaps (except the first posting),
    then encoded with Variable-Byte Encoding into a bytestream.

    Example:
    postings list [34, 67, 89, 454] is first converted to gap-based form,
    i.e., [34, 33, 22, 365], then encoded with Variable-Byte Encoding
    into a bytestream.

    ASSUMPTION: postings_list for a term fits in memory.

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encode a number using Variable-Byte Encoding.
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend to front
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # set continuation bit on the last byte
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Encode (with compression) a list of numbers
        using Variable-Byte Encoding.
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a byte stream (with Variable-Byte
        Encoding). Convert to gap-based list before encoding.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            Bytearray representing integer order in postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode term-frequency list into a byte stream.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            Bytearray representing raw TF values of term occurrences
            for each document in the postings list
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decode a bytestream previously encoded
        with variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list from a byte stream.
        Note that decoded bytestream is still in gap-based form.

        Parameters
        ----------
        encoded_postings_list: bytes
            Bytearray representing encoded postings list produced
            by static method encode above.

        Returns
        -------
        List[int]
            List of docIDs decoded from encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode term-frequency list from a byte stream.

        Parameters
        ----------
        encoded_tf_list: bytes
            Bytearray representing encoded term frequencies list produced
            by static method encode_tf above.

        Returns
        -------
        List[int]
            List of term frequencies decoded from encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)



class EliasGammaPostings:
    """ 
    Elias-Gamma postings: docIDs are stored as gap list and then encoded
    with Elias-Gamma. TF list values are encoded directly.

    Since Elias-Gamma is bit-based, bytestream stores:
    - first byte: number of 0-padding bits at payload end (0..7)
    - next bytes: gamma-encoded payload bitstream

    ASSUMPTION: all encoded numbers are non-negative integers (>= 0).
    Value 0 is handled via shift transform: encode(n + 1), decode(m - 1).
    """

    @staticmethod
    def gamma_encode_number(number):
        """Encode one positive integer with Elias-Gamma."""
        if number <= 0:
            raise ValueError(f"Elias-Gamma only supports positive integers, got {number}")
        binary = bin(number)[2:]
        prefix = '0' * (len(binary) - 1)
        return prefix + binary

    @staticmethod
    def gamma_encode(list_of_numbers):
        """Encode a list of non-negative integers to Elias-Gamma bytestream."""
        bits = []
        for idx, n in enumerate(list_of_numbers):
            if n < 0:
                raise ValueError(
                    f"Elias-Gamma only supports non-negative integers, got {n} at index {idx}"
                )
            # Shift +1 so value 0 can be represented in Elias-Gamma.
            bits.append(EliasGammaPostings.gamma_encode_number(n + 1))
        bitstream = ''.join(bits)
        padding = (8 - (len(bitstream) % 8)) % 8
        bitstream += '0' * padding

        payload_bytes = bytearray()
        for i in range(0, len(bitstream), 8):
            payload_bytes.append(int(bitstream[i:i + 8], 2))

        return bytes([padding]) + bytes(payload_bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a byte stream (with Elias-Gamma).
        Postings are first converted to a gap-based list.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            Bytearray representing integer order in postings_list
        """
        if not postings_list:
            return EliasGammaPostings.gamma_encode([])

        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return EliasGammaPostings.gamma_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode term-frequency list into a byte stream.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            Bytearray representing raw TF values of term occurrences
            for each document in the postings list
        """
        return EliasGammaPostings.gamma_encode(tf_list)

    @staticmethod
    def gamma_decode(encoded_bytestream):
        """Decode bytestream Elias-Gamma menjadi list of non-negative integers."""
        if not encoded_bytestream:
            return []

        padding = encoded_bytestream[0]
        payload = encoded_bytestream[1:]
        if padding < 0 or padding > 7:
            raise ValueError("Invalid Elias-Gamma padding")

        bitstream = ''.join(format(byte, '08b') for byte in payload)
        if padding:
            bitstream = bitstream[:-padding]

        numbers = []
        i = 0
        n_bits = len(bitstream)
        while i < n_bits:
            zeros = 0
            while i < n_bits and bitstream[i] == '0':
                zeros += 1
                i += 1

            if i >= n_bits:
                break

            if i + zeros >= n_bits:
                raise ValueError("Truncated Elias-Gamma bitstream")

            number_bits = bitstream[i:i + zeros + 1]
            numbers.append(int(number_bits, 2))
            i += zeros + 1

        # Inverse transform of shift +1 used during encoding.
        return [n - 1 for n in numbers]

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list from a byte stream.
        Note that decoded bytestream is still in gap-based form.

        Parameters
        ----------
        encoded_postings_list: bytes
            Bytearray representing encoded postings list produced
            by static method encode above.

        Returns
        -------
        List[int]
            List of docIDs decoded from encoded_postings_list
        """
        decoded_postings_list = EliasGammaPostings.gamma_decode(encoded_postings_list)
        if not decoded_postings_list:
            return []
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode term-frequency list from a byte stream.

        Parameters
        ----------
        encoded_tf_list: bytes
            Bytearray representing encoded term frequencies list produced
            by static method encode_tf above.

        Returns
        -------
        List[int]
            List of term frequencies decoded from encoded_tf_list
        """
        return EliasGammaPostings.gamma_decode(encoded_tf_list)


class VBEPostingsEliasGammaTF:
    """
    Hybrid codec:
    - Postings list (docID gaps) uses Variable-Byte Encoding
    - TF list uses Elias-Gamma Encoding
    """

    @staticmethod
    def encode(postings_list):
        return VBEPostings.encode(postings_list)

    @staticmethod
    def decode(encoded_postings_list):
        return VBEPostings.decode(encoded_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        return EliasGammaPostings.encode_tf(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        return EliasGammaPostings.decode_tf(encoded_tf_list)


class EliasGammaPostingsVBETF:
    """
    Hybrid codec:
    - Postings list (docID gaps) uses Elias-Gamma Encoding
    - TF list uses Variable-Byte Encoding
    """

    @staticmethod
    def encode(postings_list):
        return EliasGammaPostings.encode(postings_list)

    @staticmethod
    def decode(encoded_postings_list):
        return EliasGammaPostings.decode(encoded_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        return VBEPostings.encode_tf(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        return VBEPostings.decode_tf(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [
        StandardPostings,
        VBEPostings,
        EliasGammaPostings,
        VBEPostingsEliasGammaTF,
        EliasGammaPostingsVBETF,
    ]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("encoded postings bytes: ", encoded_postings_list)
        print("encoded postings size : ", len(encoded_postings_list), "bytes")
        print("encoded TF list bytes : ", encoded_tf_list)
        print("encoded TF size       : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("decoded postings: ", decoded_posting_list)
        print("decoded TF list : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "decoded output does not match original postings"
        assert decoded_tf_list == tf_list, "decoded output does not match original TF list"
        print()
