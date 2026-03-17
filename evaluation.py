import re
import math
from bsbi import BSBIIndex
from compression import VBEPostingsEliasGammaTF
from tqdm import tqdm

######## >>>>> an IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """Compute search effectiveness score using
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         binary vector such as [1, 0, 1, 1, 1, 0]
         gold-standard relevance labels for docs at rank 1, 2, 3, etc.
         Example: [1, 0, 1, 1, 1, 0] means the doc at rank-1 is relevant,
           rank-2 is not relevant, ranks 3/4/5 are relevant, and
           rank-6 is not relevant
        
      Returns
      -------
      Float
        RBP score
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


def dcg(ranking):
  """Compute Discounted Cumulative Gain (DCG) for a binary/graded ranking."""
  score = 0.0
  for i, rel in enumerate(ranking, start=1):
    score += (2 ** rel - 1) / math.log2(i + 1)
  return score


def ndcg(ranking):
  """Compute Normalized DCG (NDCG)."""
  ideal = sorted(ranking, reverse=True)
  ideal_dcg = dcg(ideal)
  if ideal_dcg == 0:
    return 0.0
  return dcg(ranking) / ideal_dcg


def ap(ranking):
  """Compute Average Precision (AP)."""
  num_relevant = sum(ranking)
  if num_relevant == 0:
    return 0.0

  precision_sum = 0.0
  relevant_so_far = 0
  for i, rel in enumerate(ranking, start=1):
    if rel:
      relevant_so_far += 1
      precision_sum += relevant_so_far / i
  return precision_sum / num_relevant


######## >>>>> load qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """Load query relevance judgments (qrels)
      as a dictionary-of-dictionaries
      qrels[query id][document id]

      where, for example, qrels["Q3"][12] = 1 means Doc 12
      is relevant to Q3; and qrels["Q3"][10] = 0 means
      Doc 10 is not relevant to Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUATION!

def extract_doc_id(doc_path):
  """Extract numeric document id from a path like ./collection/1/123.txt."""
  match = re.search(r'/([^/]+)\.txt$', doc_path)
  if match is None:
    raise ValueError(f"Cannot parse doc id from path: {doc_path}")
  return int(match.group(1))

def eval(qrels, query_file = "queries.txt", k = 1000):
  """ 
    Iterate over all 30 queries and compute mean RBP for
    both TF-IDF and BM25 over top-k retrieved documents.
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostingsEliasGammaTF, \
                          output_dir = 'index')

  with open(query_file) as file:
    query_lines = file.readlines()
    rbp_scores_tfidf = []
    dcg_scores_tfidf = []
    ndcg_scores_tfidf = []
    ap_scores_tfidf = []

    rbp_scores_bm25 = []
    dcg_scores_bm25 = []
    ndcg_scores_bm25 = []
    ap_scores_bm25 = []

    for qline in tqdm(query_lines, desc="Evaluating queries"):
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # Be careful: doc IDs from indexing may differ from the IDs listed in qrels.
      ranking_tfidf = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k):
          did = extract_doc_id(doc)
          ranking_tfidf.append(qrels[qid][did])
          rbp_scores_tfidf.append(rbp(ranking_tfidf))
          dcg_scores_tfidf.append(dcg(ranking_tfidf))
          ndcg_scores_tfidf.append(ndcg(ranking_tfidf))
          ap_scores_tfidf.append(ap(ranking_tfidf))

      ranking_bm25 = []
      for (score, doc) in BSBI_instance.retrieve_bm25(query, k = k):
          did = extract_doc_id(doc)
          ranking_bm25.append(qrels[qid][did])
          rbp_scores_bm25.append(rbp(ranking_bm25))
          dcg_scores_bm25.append(dcg(ranking_bm25))
          ndcg_scores_bm25.append(ndcg(ranking_bm25))
          ap_scores_bm25.append(ap(ranking_bm25))

  print("Evaluation results over 30 queries")
  print("TF-IDF RBP  =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
  print("TF-IDF DCG  =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
  print("TF-IDF NDCG =", sum(ndcg_scores_tfidf) / len(ndcg_scores_tfidf))
  print("TF-IDF AP   =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))
  print("BM25   RBP  =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
  print("BM25   DCG  =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
  print("BM25   NDCG =", sum(ndcg_scores_bm25) / len(ndcg_scores_bm25))
  print("BM25   AP   =", sum(ap_scores_bm25) / len(ap_scores_bm25))

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels is incorrect"
  assert qrels["Q1"][300] == 0, "qrels is incorrect"

  eval(qrels)