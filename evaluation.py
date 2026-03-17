import re
import math
import time
import argparse
from pathlib import Path

import numpy as np
from bsbi import BSBIIndex
from compression import VBEPostingsEliasGammaTF
from lsi_faiss import load_lsi
from sklearn.preprocessing import normalize
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

def eval(qrels, query_file = "queries.txt", k = 1000, use_patricia = False, lsi_output_dir = None):
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

    rbp_scores_bm25_wand = []
    dcg_scores_bm25_wand = []
    ndcg_scores_bm25_wand = []
    ap_scores_bm25_wand = []

    rbp_scores_lsi = []
    dcg_scores_lsi = []
    ndcg_scores_lsi = []
    ap_scores_lsi = []

    total_time_bm25 = 0.0
    total_time_bm25_wand = 0.0
    total_time_lsi = 0.0

    lsi_ready = lsi_output_dir is not None and Path(lsi_output_dir).exists()
    if lsi_ready:
      lsi_index, lsi_vectorizer, lsi_svd, lsi_meta = load_lsi(lsi_output_dir)

    for qline in tqdm(query_lines, desc="Evaluating queries"):
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # Be careful: doc IDs from indexing may differ from the IDs listed in qrels.
      ranking_tfidf = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k, use_patricia = use_patricia):
          did = extract_doc_id(doc)
          ranking_tfidf.append(qrels[qid][did])
          rbp_scores_tfidf.append(rbp(ranking_tfidf))
          dcg_scores_tfidf.append(dcg(ranking_tfidf))
          ndcg_scores_tfidf.append(ndcg(ranking_tfidf))
          ap_scores_tfidf.append(ap(ranking_tfidf))

      ranking_bm25 = []
      start_bm25 = time.perf_counter()
      results_bm25 = BSBI_instance.retrieve_bm25(query, k = k, use_patricia = use_patricia)
      total_time_bm25 += (time.perf_counter() - start_bm25)
      for (score, doc) in results_bm25:
          did = extract_doc_id(doc)
          ranking_bm25.append(qrels[qid][did])
          rbp_scores_bm25.append(rbp(ranking_bm25))
          dcg_scores_bm25.append(dcg(ranking_bm25))
          ndcg_scores_bm25.append(ndcg(ranking_bm25))
          ap_scores_bm25.append(ap(ranking_bm25))

      ranking_bm25_wand = []
      start_bm25_wand = time.perf_counter()
      results_bm25_wand = BSBI_instance.retrieve_bm25_wand(query, k = k, use_patricia = use_patricia)
      total_time_bm25_wand += (time.perf_counter() - start_bm25_wand)
      for (score, doc) in results_bm25_wand:
          did = extract_doc_id(doc)
          ranking_bm25_wand.append(qrels[qid][did])
          rbp_scores_bm25_wand.append(rbp(ranking_bm25_wand))
          dcg_scores_bm25_wand.append(dcg(ranking_bm25_wand))
          ndcg_scores_bm25_wand.append(ndcg(ranking_bm25_wand))
          ap_scores_bm25_wand.append(ap(ranking_bm25_wand))

      if lsi_ready:
        ranking_lsi = []
        start_lsi = time.perf_counter()
        q_tfidf = lsi_vectorizer.transform([query])
        q_lsi = lsi_svd.transform(q_tfidf).astype(np.float32)
        q_lsi = normalize(q_lsi, norm="l2", copy=False)
        _, doc_ids = lsi_index.search(q_lsi, k)
        total_time_lsi += (time.perf_counter() - start_lsi)

        for doc_id in doc_ids[0]:
          if doc_id < 0:
            continue
          did = extract_doc_id(lsi_meta["doc_paths"][int(doc_id)])
          ranking_lsi.append(qrels[qid][did])
          rbp_scores_lsi.append(rbp(ranking_lsi))
          dcg_scores_lsi.append(dcg(ranking_lsi))
          ndcg_scores_lsi.append(ndcg(ranking_lsi))
          ap_scores_lsi.append(ap(ranking_lsi))

  def fmt_num(value):
    # Keep output concise with up to 4 decimals while avoiding trailing zeros.
    text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".")

  def print_aligned(rows):
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
      print(f"{label:<{width}} = {fmt_num(value)}")

  print("Evaluation results over 30 queries")
  print_aligned([
    ("TF-IDF RBP", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf)),
    ("TF-IDF DCG", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf)),
    ("TF-IDF NDCG", sum(ndcg_scores_tfidf) / len(ndcg_scores_tfidf)),
    ("TF-IDF AP", sum(ap_scores_tfidf) / len(ap_scores_tfidf)),
    ("BM25 RBP", sum(rbp_scores_bm25) / len(rbp_scores_bm25)),
    ("BM25 DCG", sum(dcg_scores_bm25) / len(dcg_scores_bm25)),
    ("BM25 NDCG", sum(ndcg_scores_bm25) / len(ndcg_scores_bm25)),
    ("BM25 AP", sum(ap_scores_bm25) / len(ap_scores_bm25)),
    ("BM25+WAND RBP", sum(rbp_scores_bm25_wand) / len(rbp_scores_bm25_wand)),
    ("BM25+WAND DCG", sum(dcg_scores_bm25_wand) / len(dcg_scores_bm25_wand)),
    ("BM25+WAND NDCG", sum(ndcg_scores_bm25_wand) / len(ndcg_scores_bm25_wand)),
    ("BM25+WAND AP", sum(ap_scores_bm25_wand) / len(ap_scores_bm25_wand)),
  ])
  if lsi_ready and rbp_scores_lsi:
    print_aligned([
      ("LSI+FAISS RBP", sum(rbp_scores_lsi) / len(rbp_scores_lsi)),
      ("LSI+FAISS DCG", sum(dcg_scores_lsi) / len(dcg_scores_lsi)),
      ("LSI+FAISS NDCG", sum(ndcg_scores_lsi) / len(ndcg_scores_lsi)),
      ("LSI+FAISS AP", sum(ap_scores_lsi) / len(ap_scores_lsi)),
    ])

  n_queries = len(query_lines)
  print("\nRetrieval time comparison (30 queries)")
  print_aligned([
    ("BM25 brute-force total (s)", total_time_bm25),
    ("BM25 WAND total (s)", total_time_bm25_wand),
    ("BM25 brute-force avg/query (s)", total_time_bm25 / n_queries),
    ("BM25 WAND avg/query (s)", total_time_bm25_wand / n_queries),
  ])
  if lsi_ready:
    print_aligned([
      ("LSI+FAISS total (s)", total_time_lsi),
      ("LSI+FAISS avg/query (s)", total_time_lsi / n_queries),
    ])
  if total_time_bm25_wand > 0:
    print_aligned([
      ("Speedup (brute/WAND)", total_time_bm25 / total_time_bm25_wand),
    ])
  if lsi_ready and total_time_lsi > 0:
    print_aligned([
      ("Speedup (BM25/LSI)", total_time_bm25 / total_time_lsi),
      ("Speedup (WAND/LSI)", total_time_bm25_wand / total_time_lsi),
    ])

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Evaluate retrieval metrics over all queries")
  parser.add_argument("--qrels", default="qrels.txt", help="Path to qrels file")
  parser.add_argument("--queries", default="queries.txt", help="Path to queries file")
  parser.add_argument("--k", type=int, default=10, help="Top-k retrieved documents per query")
  parser.add_argument(
    "--use-patricia",
    action="store_true",
    help="Use Patricia tree for query term lookup during retrieval",
  )
  parser.add_argument(
    "--lsi-output-dir",
    default="lsi_index",
    help="Directory containing LSI+FAISS artifacts (set empty to disable)",
  )
  args = parser.parse_args()

  qrels = load_qrels(qrel_file=args.qrels)

  assert qrels["Q1"][166] == 1, "qrels is incorrect"
  assert qrels["Q1"][300] == 0, "qrels is incorrect"

  lsi_output_dir = args.lsi_output_dir.strip() or None
  eval(
    qrels,
    query_file=args.queries,
    k=args.k,
    use_patricia=args.use_patricia,
    lsi_output_dir=lsi_output_dir,
  )