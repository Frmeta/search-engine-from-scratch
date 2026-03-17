import argparse
from threading import Lock
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from adaptive_retrieval import AdaptiveRetriever
from bsbi import BSBIIndex, SPIMIIndex
from compression import VBEPostingsEliasGammaTF
from lsi_faiss import query_lsi

app = Flask(__name__)

INDEX_INSTANCE = None
ADAPTIVE_RETRIEVER = None
ADAPTIVE_ERROR = None
INIT_LOCK = Lock()

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Search Engine UI</title>
  <style>
    :root {
      --bg: #f7f4ee;
      --ink: #1b1b1b;
      --panel: #fffdfa;
      --accent: #0a7f5a;
      --line: #ded7ca;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: radial-gradient(circle at 20% 0%, #fff8eb 0%, var(--bg) 45%), var(--bg);
    }
    .wrap {
      max-width: 1080px;
      margin: 40px auto;
      padding: 0 20px 40px;
    }
    .hero {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.05);
    }
    h1 {
      margin: 0 0 12px;
      font-size: 2rem;
      letter-spacing: 0.2px;
    }
    .sub {
      margin: 0 0 20px;
      color: #4f4a42;
    }
    form {
      display: grid;
      gap: 12px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }
    input[type="text"], input[type="number"] {
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 10px;
      font-size: 1rem;
      background: #fff;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 12px 18px;
      cursor: pointer;
      font-size: 1rem;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
    }
    .opts {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 4px;
    }
    label {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      font-size: 0.95rem;
    }
    .results {
      margin-top: 22px;
      display: grid;
      gap: 14px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }
    .card h2 {
      margin: 0 0 10px;
      font-size: 1.1rem;
    }
    .hit {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 8px 0;
      border-top: 1px dashed #e9e3d9;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.9rem;
    }
    .hit:first-child { border-top: 0; }
    .warn {
      margin-top: 12px;
      color: #8a4e00;
      font-size: 0.92rem;
    }
    .doc-btn {
      border: 0;
      background: transparent;
      color: #185a45;
      text-align: left;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.9rem;
      padding: 0;
      cursor: pointer;
      text-decoration: underline;
    }
    .doc-modal {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.45);
      padding: 18px;
      z-index: 999;
    }
    .doc-modal.open { display: flex; }
    .doc-panel {
      width: min(900px, 100%);
      max-height: 80vh;
      overflow: auto;
      background: #fff;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 16px;
    }
    .doc-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .doc-content {
      white-space: pre-wrap;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.9rem;
      line-height: 1.45;
    }
    @media (max-width: 768px) {
      .row { grid-template-columns: 1fr; }
      button { width: 100%; }
      h1 { font-size: 1.6rem; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Search Engine From Scratch</h1>
      <p class="sub">Choose one retrieval method per search: TF-IDF, BM25, or LSI+FAISS.</p>
      <form id="searchForm">
        <div class="row">
          <input id="query" name="query" type="text" placeholder="Type your query..." required />
          <button type="submit">Search</button>
        </div>
        <div class="opts">
          <label><input id="usePatricia" type="checkbox" /> Use Patricia lookup</label>
          <label>Method
            <select id="method" style="padding:8px;border:1px solid var(--line);border-radius:8px;background:#fff;">
              <option value="bm25" selected>BM25</option>
              <option value="tfidf">TF-IDF</option>
              <option value="lsi">LSI+FAISS</option>
            </select>
          </label>
          <label>Top-K <input id="topk" type="number" min="1" max="50" value="10" style="width:80px" /></label>
        </div>
      </form>
      <div id="warning" class="warn"></div>
    </section>

    <section id="results" class="results"></section>
  </div>

  <div id="docModal" class="doc-modal" role="dialog" aria-modal="true">
    <div class="doc-panel">
      <div class="doc-head">
        <strong id="docTitle">Document</strong>
        <button id="closeDoc" type="button">Close</button>
      </div>
      <div id="docBody" class="doc-content"></div>
    </div>
  </div>

  <script>
    const form = document.getElementById('searchForm');
    const resultsEl = document.getElementById('results');
    const warningEl = document.getElementById('warning');
    const docModal = document.getElementById('docModal');
    const docTitle = document.getElementById('docTitle');
    const docBody = document.getElementById('docBody');
    const closeDocBtn = document.getElementById('closeDoc');

    function closeDocModal() {
      docModal.classList.remove('open');
      docBody.textContent = '';
    }

    async function openDocument(docPath) {
      docTitle.textContent = docPath;
      docBody.textContent = 'Loading...';
      docModal.classList.add('open');

      const res = await fetch(`/document?path=${encodeURIComponent(docPath)}`);
      const data = await res.json();
      if (!res.ok) {
        docBody.textContent = data.error || 'Failed to load document';
        return;
      }
      docBody.textContent = data.content;
    }

    closeDocBtn.addEventListener('click', closeDocModal);
    docModal.addEventListener('click', (e) => {
      if (e.target === docModal) closeDocModal();
    });

    function renderBucket(hits) {
      const card = document.createElement('article');
      card.className = 'card';
      if (!hits || hits.length === 0) {
        const empty = document.createElement('div');
        empty.textContent = 'No results';
        card.appendChild(empty);
        return card;
      }
      hits.forEach((h) => {
        const row = document.createElement('div');
        row.className = 'hit';
        row.innerHTML = `<button class="doc-btn" type="button">${h.doc}</button><strong>${h.score.toFixed(4)}</strong>`;
        const docBtn = row.querySelector('.doc-btn');
        docBtn.addEventListener('click', () => openDocument(h.doc));
        card.appendChild(row);
      });
      return card;
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      resultsEl.innerHTML = '';
      warningEl.textContent = '';

      const payload = {
        query: document.getElementById('query').value,
        topk: Number(document.getElementById('topk').value || 10),
        use_patricia: document.getElementById('usePatricia').checked,
        method: document.getElementById('method').value,
      };

      const res = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        warningEl.textContent = data.error || 'Search request failed';
        return;
      }

      if (data.warnings && data.warnings.length > 0) {
        warningEl.textContent = data.warnings.join(' | ');
      }

      resultsEl.appendChild(renderBucket(data.results));
    });
  </script>
</body>
</html>
"""


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _format_hits(hits):
    return [{"score": float(score), "doc": str(doc)} for score, doc in hits]


def _resolve_doc_path(path_str):
  if not path_str:
    raise ValueError("Missing document path")

  raw = Path(path_str.strip())
  if raw.is_absolute():
    candidate = raw
  else:
    candidate = (Path.cwd() / raw).resolve()

  allowed_root = (Path.cwd() / "collection").resolve()
  if not str(candidate).startswith(str(allowed_root)):
    raise ValueError("Document path is outside collection directory")

  if not candidate.is_file():
    raise ValueError("Document not found")

  return candidate


def initialize_engine(spimi=False, adaptive_index_dir="pt_index"):
    global INDEX_INSTANCE, ADAPTIVE_RETRIEVER, ADAPTIVE_ERROR

    with INIT_LOCK:
        if INDEX_INSTANCE is not None:
            return

        index_cls = SPIMIIndex if spimi else BSBIIndex
        INDEX_INSTANCE = index_cls(
            data_dir="collection",
            postings_encoding=VBEPostingsEliasGammaTF,
            output_dir="index",
        )

        try:
            ADAPTIVE_RETRIEVER = AdaptiveRetriever(
                collection_dir="collection",
                index_dir=adaptive_index_dir,
            )
            ADAPTIVE_ERROR = None
        except RuntimeError as exc:
            ADAPTIVE_RETRIEVER = None
            ADAPTIVE_ERROR = str(exc)


@app.get("/")
def index_page():
    return render_template_string(HTML_PAGE)


@app.post("/search")
def search_api():
    body = request.get_json(silent=True) or request.form
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query must not be empty"}), 400

    topk = int(body.get("topk", 10) or 10)
    topk = max(1, min(topk, 50))

    use_patricia = _to_bool(body.get("use_patricia"), default=False)
    method = (body.get("method") or "bm25").strip().lower()
    method_labels = {
      "tfidf": "TF-IDF",
      "bm25": "BM25",
      "lsi": "LSI+FAISS",
    }
    if method not in method_labels:
      return jsonify({"error": "Invalid method. Use one of: tfidf, bm25, lsi"}), 400

    warnings = []

    hits = []
    if method == "tfidf":
      hits = INDEX_INSTANCE.retrieve_tfidf(query, k=topk, use_patricia=use_patricia)
    elif method == "bm25":
      hits = INDEX_INSTANCE.retrieve_bm25(query, k=topk, use_patricia=use_patricia)
    else:
      try:
        hits = query_lsi(args.lsi_output_dir, query, topk=topk)
      except Exception as exc:
        warnings.append(f"LSI+FAISS failed: {exc}")
        hits = []

    return jsonify(
        {
            "query": query,
            "topk": topk,
          "method": method,
          "method_label": method_labels[method],
            "warnings": warnings,
          "results": _format_hits(hits),
        }
    )


@app.get("/document")
def document_api():
    path_str = request.args.get("path", "")
    try:
        doc_path = _resolve_doc_path(path_str)
        content = doc_path.read_text(encoding="utf8", errors="ignore")
        return jsonify({"path": str(doc_path), "content": content})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to read document: {exc}"}), 500


def build_args():
    parser = argparse.ArgumentParser(description="Flask frontend for the search engine")
    parser.add_argument("--spimi", action="store_true", help="Use SPIMI index class")
    parser.add_argument(
        "--adaptive-index-dir",
        default="pt_index",
        help="Directory for adaptive retrieval index artifacts",
    )
    parser.add_argument(
        "--lsi-output-dir",
        default="lsi_index",
        help="Directory for LSI+FAISS artifacts",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    initialize_engine(spimi=args.spimi, adaptive_index_dir=args.adaptive_index_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)
