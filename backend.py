"""
╔═══════════════════════════════════════════════════════════════╗
║          THE EXPLORER — Backend Search Engine                 ║
║  FastAPI + Sentence-Transformers + BM25 + Neural Re-ranking  ║
╚═══════════════════════════════════════════════════════════════╝

Stack:
  - FastAPI            → REST API server
  - sentence-transformers → Neural embeddings (all-MiniLM-L6-v2, ~22M params)
  - BM25               → Lexical retrieval (first-pass)
  - FAISS              → Approximate nearest-neighbour vector search
  - BeautifulSoup      → Web page crawler/parser
  - SQLite             → Document store + metadata
  - NLTK WordNet       → Query expansion (semantic synonyms)
  - CORS               → Works with the HTML frontend

Install:
  pip install fastapi uvicorn sentence-transformers faiss-cpu \
              beautifulsoup4 requests rank_bm25 numpy aiohttp nltk

Run:
  uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import urllib.parse
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

import aiohttp
import numpy as np
import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── WordNet query expansion ──
try:
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
    print("[Explorer] WordNet loaded — query expansion enabled.")
except Exception as _e:
    WORDNET_AVAILABLE = False
    print(f"[Explorer] WordNet not available ({_e}), query expansion disabled.")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # ~22M params, fast CPU inference

DB_PATH   = "explorer.db"
MAX_CRAWL = 50          # max pages per crawl job
RESULTS_N = 10          # results per page

# ─────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="The Explorer Search API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT UNIQUE,
            title       TEXT,
            body        TEXT,
            snippet     TEXT,
            domain      TEXT,
            crawled_at  TEXT,
            doc_hash    TEXT
        );

        CREATE TABLE IF NOT EXISTS search_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            query     TEXT,
            results   INTEGER,
            latency   REAL,
            ts        TEXT
        );

        CREATE TABLE IF NOT EXISTS crawl_queue (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            url     TEXT UNIQUE,
            status  TEXT DEFAULT 'pending',
            depth   INTEGER DEFAULT 0
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS fts
            USING fts5(title, body, url UNINDEXED, content='documents', content_rowid='id');
    """)
    con.commit()
    con.close()

def get_con():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# ─────────────────────────────────────────────────────────────
# NEURAL MODEL
# ─────────────────────────────────────────────────────────────
print(f"[Explorer] Loading embedding model: {MODEL_NAME} …")
embedder = SentenceTransformer(MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()
print(f"[Explorer] Model loaded. Embedding dim: {EMBED_DIM}")

try:
    import faiss
    index = faiss.IndexFlatIP(EMBED_DIM)
    USE_FAISS = True
    print("[Explorer] FAISS index ready.")
except ImportError:
    USE_FAISS = False
    print("[Explorer] FAISS not available, using numpy brute-force.")

# In-memory stores (rebuilt on startup from DB)
doc_store: List[dict] = []
doc_embeddings: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────
# TEXT UTILS
# ─────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "to","of","in","for","on","with","as","by","at","from","and","or","but",
    "not","that","this","these","those","it","its","they","their","there",
}

def tokenize(text: str) -> List[str]:
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def expand_query(query: str) -> str:
    """
    Expand query with WordNet synonyms for entity/semantic awareness.
    e.g. 'cat'       -> 'cat feline kitty felid'
         'aeroplane' -> 'aeroplane airplane aircraft plane jet'
         'house'     -> 'house home dwelling residence abode'
    """
    if not WORDNET_AVAILABLE:
        return query
    tokens = query.lower().split()
    expanded = list(tokens)
    for token in tokens:
        syns = set()
        for syn in wn.synsets(token)[:3]:        # top 3 senses only
            for lemma in syn.lemmas()[:4]:        # top 4 lemmas per sense
                word = lemma.name().replace("_", " ")
                if word.lower() != token and len(word) > 2:
                    syns.add(word.lower())
        expanded.extend(list(syns)[:6])           # cap at 6 synonyms per token
    # Deduplicate while preserving order
    seen = set()
    result = []
    for w in expanded:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return " ".join(result)


def make_snippet(body: str, query: str, length: int = 200) -> str:
    """Extract a relevant snippet around query terms."""
    terms = tokenize(query)
    lower = body.lower()
    best_pos = 0
    best_score = 0
    for i in range(0, max(1, len(body) - length), 30):
        chunk = lower[i:i+length]
        score = sum(chunk.count(t) for t in terms)
        if score > best_score:
            best_score = score
            best_pos = i
    raw = body[best_pos:best_pos+length].strip()
    return ("…" if best_pos > 0 else "") + raw + ("…" if best_pos + length < len(body) else "")


# ─────────────────────────────────────────────────────────────
# INDEX BUILDER
# ─────────────────────────────────────────────────────────────
bm25: Optional[BM25Okapi] = None

def rebuild_index():
    """Load all docs from DB, build BM25 + FAISS index."""
    global doc_store, doc_embeddings, bm25, index

    con = get_con()
    rows = con.execute(
        "SELECT id, url, title, body, snippet, domain, crawled_at FROM documents"
    ).fetchall()
    con.close()

    if not rows:
        print("[Explorer] No documents in index yet.")
        return

    doc_store = [
        {"id": r[0], "url": r[1], "title": r[2], "body": r[3],
         "snippet": r[4], "domain": r[5], "crawled_at": r[6]}
        for r in rows
    ]

    # BM25 — tokenize with expanded terms so synonyms are indexed too
    tokenized = [tokenize(d["title"] + " " + d["body"]) for d in doc_store]
    bm25 = BM25Okapi(tokenized)

    # Neural embeddings
    texts = [d["title"] + ". " + d["body"][:512] for d in doc_store]
    embs = embedder.encode(texts, batch_size=32, show_progress_bar=True,
                           normalize_embeddings=True)
    doc_embeddings = np.array(embs, dtype=np.float32)

    if USE_FAISS:
        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(doc_embeddings)

    print(f"[Explorer] Index built: {len(doc_store)} documents.")


# ─────────────────────────────────────────────────────────────
# CRAWLER  (HTML — for non-Wikipedia sites)
# ─────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": "TheExplorer/1.0 (educational search engine)"}

def extract_page(url: str, html: str) -> dict:
    """Parse HTML into structured document."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","nav","footer","header","aside"]):
        tag.decompose()

    title = (soup.find("title") or soup.find("h1") or soup.find("h2"))
    title = title.get_text(strip=True) if title else url

    meta_desc = soup.find("meta", attrs={"name": "description"})
    snippet = meta_desc["content"] if meta_desc and meta_desc.get("content") else ""

    paras = soup.find_all(["p","article","section","main"])
    body  = " ".join(p.get_text(" ", strip=True) for p in paras)
    body  = re.sub(r"\s+", " ", body).strip()[:8000]

    if not snippet and body:
        snippet = body[:200] + "…"

    links = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(url, a["href"])
        if href.startswith("http") and urllib.parse.urlparse(href).netloc:
            links.append(href)

    domain = urllib.parse.urlparse(url).netloc
    return {"title": title, "body": body, "snippet": snippet,
            "domain": domain, "links": links[:50]}


async def crawl_url(session: aiohttp.ClientSession, url: str) -> Optional[dict]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8),
                               headers=HEADERS, ssl=False) as resp:
            if resp.status != 200:
                return None
            ct = resp.headers.get("content-type","")
            if "text/html" not in ct:
                return None
            html = await resp.text(errors="ignore")
            return extract_page(url, html)
    except Exception as e:
        print(f"[Crawl] Error {url}: {e}")
        return None


async def run_crawl_job(seed_urls: List[str], max_pages: int = MAX_CRAWL):
    """BFS crawler starting from seed URLs."""
    visited = set()
    queue   = list(seed_urls)
    crawled = 0

    con = get_con()

    async with aiohttp.ClientSession() as session:
        while queue and crawled < max_pages:
            url = queue.pop(0)
            if url in visited or len(url) > 500:
                continue
            visited.add(url)

            page = await crawl_url(session, url)
            if not page or len(page["body"]) < 100:
                continue

            doc_hash = hashlib.md5(page["body"].encode()).hexdigest()

            try:
                con.execute("""
                    INSERT OR REPLACE INTO documents
                    (url, title, body, snippet, domain, crawled_at, doc_hash)
                    VALUES (?,?,?,?,?,?,?)
                """, (url, page["title"], page["body"], page["snippet"],
                      page["domain"], datetime.utcnow().isoformat(), doc_hash))
                con.commit()
                crawled += 1
                print(f"[Crawl] {crawled}/{max_pages} — {page['title'][:60]}")
            except Exception as e:
                print(f"[Crawl] DB error: {e}")

            # Enqueue links (same domain priority)
            base_domain = urllib.parse.urlparse(url).netloc
            for link in page["links"]:
                if link not in visited:
                    link_domain = urllib.parse.urlparse(link).netloc
                    if link_domain == base_domain:
                        queue.insert(0, link)   # same domain → front
                    else:
                        queue.append(link)

    con.close()
    rebuild_index()
    print(f"[Crawl] Job complete. {crawled} pages crawled.")


# ─────────────────────────────────────────────────────────────
# SEARCH ENGINE CORE
# ─────────────────────────────────────────────────────────────
def hybrid_search(query: str, top_k: int = RESULTS_N, page: int = 1) -> dict:
    """
    Two-stage retrieval:
      1. BM25 lexical retrieval on query-expanded tokens
      2. Neural re-ranking with cosine similarity
      3. Reciprocal Rank Fusion to blend scores
    """
    t0 = time.perf_counter()

    if not doc_store:
        return {"results": [], "total": 0, "query": query,
                "latency": 0, "page": page, "pages": 0}

    # Expand query with synonyms before BM25 tokenisation
    expanded_query = expand_query(query)
    q_tokens = tokenize(expanded_query)
    offset   = (page - 1) * top_k

    # ── Stage 1: BM25 ──
    bm25_scores = np.zeros(len(doc_store))
    if bm25 and q_tokens:
        raw = bm25.get_scores(q_tokens)
        bm25_scores = raw / (raw.max() + 1e-9)   # normalise to [0,1]

    # ── Stage 2: Neural ──
    neural_scores = np.zeros(len(doc_store))
    if doc_embeddings is not None:
        q_emb = embedder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        if USE_FAISS:
            sims, idxs = index.search(q_emb, min(100, len(doc_store)))
            for rank, (sim, idx) in enumerate(zip(sims[0], idxs[0])):
                neural_scores[idx] = float(sim)
        else:
            sims = (doc_embeddings @ q_emb.T).flatten()
            neural_scores = sims

    # ── Stage 3: Reciprocal Rank Fusion ──
    k = 60
    bm25_ranks   = np.argsort(-bm25_scores)
    neural_ranks = np.argsort(-neural_scores)

    rrf = np.zeros(len(doc_store))
    for r, idx in enumerate(bm25_ranks):
        rrf[idx] += 1.0 / (k + r + 1)
    for r, idx in enumerate(neural_ranks):
        rrf[idx] += 1.0 / (k + r + 1)

    sorted_idx = np.argsort(-rrf)
    total      = int((rrf > 0).sum())
    page_idx   = sorted_idx[offset:offset + top_k]

    results = []
    for rank, idx in enumerate(page_idx):
        doc = doc_store[idx]
        snip = make_snippet(doc["body"], query) if doc["body"] else doc["snippet"]
        results.append({
            "rank":      offset + rank + 1,
            "title":     doc["title"],
            "url":       doc["url"],
            "domain":    doc["domain"],
            "snippet":   snip,
            "bm25":      float(bm25_scores[idx]),
            "neural":    float(neural_scores[idx]),
            "rrf":       float(rrf[idx]),
            "crawled":   doc["crawled_at"],
        })

    latency = round((time.perf_counter() - t0) * 1000, 2)  # ms

    con = get_con()
    con.execute("INSERT INTO search_log (query,results,latency,ts) VALUES (?,?,?,?)",
                (query, total, latency, datetime.utcnow().isoformat()))
    con.commit()
    con.close()

    return {
        "query":   query,
        "results": results,
        "total":   total,
        "latency": latency,
        "page":    page,
        "pages":   math.ceil(total / top_k),
    }


# ─────────────────────────────────────────────────────────────
# SEED DATA (so the engine works out-of-the-box)
# ─────────────────────────────────────────────────────────────
SEED_DOCS = [
    {"url":"https://example.com/ml-intro","title":"Introduction to Machine Learning",
     "domain":"example.com","snippet":"ML is a subset of AI that enables computers to learn from data.",
     "body":"Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed. The process begins with data observation or direct experience to look for patterns and make better decisions in the future. The primary aim is to allow computers to learn automatically without human intervention or assistance and adjust actions accordingly. Supervised learning uses labelled training data. Unsupervised learning finds hidden patterns in unlabelled data. Reinforcement learning trains agents via reward signals. Deep learning uses multilayer neural networks to learn representations of data with multiple levels of abstraction."},
    {"url":"https://example.com/neural-networks","title":"Deep Learning and Neural Networks",
     "domain":"example.com","snippet":"Deep learning uses multilayer neural networks to model complex patterns.",
     "body":"Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. The adjective deep refers to the use of multiple layers in the network. Methods used can be either supervised, semi-supervised or unsupervised. Convolutional neural networks excel at image recognition and computer vision tasks. Recurrent neural networks and transformers handle sequential data like text and audio. Backpropagation and gradient descent are the core training algorithms. Activation functions like ReLU, sigmoid and softmax introduce non-linearity. Batch normalisation and dropout are regularisation techniques."},
    {"url":"https://example.com/nlp","title":"Natural Language Processing",
     "domain":"example.com","snippet":"NLP enables computers to understand and generate human language.",
     "body":"Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. Core NLP tasks include tokenisation, part-of-speech tagging, named entity recognition, dependency parsing, coreference resolution, sentiment analysis, machine translation, and text summarisation. Transformer models like BERT, GPT, and T5 have revolutionised NLP through self-attention mechanisms and pre-training on large corpora. Word embeddings like Word2Vec and GloVe represent words as dense vectors capturing semantic relationships."},
    {"url":"https://example.com/information-retrieval","title":"Information Retrieval Systems",
     "domain":"example.com","snippet":"IR systems help users find relevant documents from large collections.",
     "body":"Information retrieval is the process of obtaining information resources relevant to an information need from a collection of those resources. The vector space model represents documents and queries as vectors in a high-dimensional term space. TF-IDF weighting reflects how important a word is to a document in a corpus. BM25 is a probabilistic retrieval function that ranks documents based on query term appearance. Dense retrieval uses neural embeddings for semantic matching beyond keyword overlap. Inverted indexes enable fast full-text search by mapping terms to posting lists of document IDs. Precision and recall are the fundamental evaluation metrics."},
    {"url":"https://example.com/transformers","title":"Transformer Architecture Explained",
     "domain":"example.com","snippet":"The transformer architecture powers modern NLP through self-attention.",
     "body":"The transformer is a deep learning model introduced in the paper Attention Is All You Need. It is based entirely on attention mechanisms and dispenses with recurrence and convolution. The architecture consists of an encoder and a decoder, each composed of a stack of identical layers. Multi-head self-attention allows the model to attend to information from different representation subspaces at different positions. Positional encoding injects information about the relative or absolute position of tokens. The transformer has become the dominant architecture for NLP tasks and is the basis for BERT, GPT, T5, and other large language models."},
    {"url":"https://example.com/python","title":"Python Programming Language",
     "domain":"example.com","snippet":"Python is a high-level general-purpose programming language.",
     "body":"Python is a high-level general-purpose programming language. Its design philosophy emphasises code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms including structured, object-oriented and functional programming. Python is often described as a batteries-included language due to its comprehensive standard library. NumPy provides fast numerical computing. Pandas enables data analysis and manipulation. Matplotlib and Seaborn handle data visualisation. Scikit-learn offers classical machine learning algorithms. PyTorch and TensorFlow power deep learning research and production."},
    {"url":"https://example.com/databases","title":"Database Systems and SQL",
     "domain":"example.com","snippet":"Database systems organise structured data for efficient retrieval.",
     "body":"A database management system is software that interacts with end users, applications, and the database itself to capture and analyse data. Relational databases store data in tables with rows and columns. SQL is the standard language for relational database management systems. Key SQL operations include SELECT for querying, INSERT for adding records, UPDATE for modifying, DELETE for removing, and JOIN for combining tables. Indexes improve query performance by enabling binary search. Transactions ensure data integrity through ACID properties: Atomicity, Consistency, Isolation, Durability. NoSQL databases like MongoDB, Redis, and Cassandra handle unstructured data and provide horizontal scalability."},
    {"url":"https://example.com/cloud","title":"Cloud Computing and Distributed Systems",
     "domain":"example.com","snippet":"Cloud computing delivers computing resources over the internet.",
     "body":"Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. Large clouds often have functions distributed over multiple locations, each location being a data centre. Infrastructure as a Service provides virtualised computing resources. Platform as a Service offers hardware and software tools over the internet. Software as a Service delivers software applications over the internet. Distributed systems coordinate multiple computers that communicate and synchronise their actions by passing messages. The CAP theorem states that a distributed system can provide only two of three guarantees: consistency, availability, and partition tolerance."},
    {"url":"https://example.com/cybersecurity","title":"Cybersecurity and Cryptography",
     "domain":"example.com","snippet":"Cybersecurity protects digital systems from attacks and breaches.",
     "body":"Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information, extorting money from users, or interrupting normal business processes. Cryptography uses mathematical techniques to secure communication. Symmetric encryption like AES uses the same key for encryption and decryption. Asymmetric encryption like RSA uses a public-private key pair. Hash functions like SHA-256 produce fixed-size digests for data integrity. Public key infrastructure manages digital certificates. Common attack vectors include phishing, SQL injection, cross-site scripting, and man-in-the-middle attacks."},
    {"url":"https://example.com/algorithms","title":"Data Structures and Algorithms",
     "domain":"example.com","snippet":"Efficient data structures and algorithms are foundational to computer science.",
     "body":"Data structures organise data in memory for efficient access and modification. Arrays provide constant-time indexed access. Linked lists enable efficient insertion and deletion. Hash tables offer average constant-time lookup. Binary search trees maintain sorted order. Heaps support priority queue operations. Graphs model relationships between entities. Algorithms define step-by-step procedures for solving computational problems. Sorting algorithms include quicksort, mergesort, and heapsort. Graph algorithms include BFS, DFS, Dijkstra and Bellman-Ford. Dynamic programming solves complex problems by breaking them into overlapping subproblems. Big O notation characterises algorithm complexity in terms of time and space."},
    {"url":"https://example.com/computer-vision","title":"Computer Vision",
     "domain":"example.com","snippet":"Computer vision enables machines to interpret visual information.",
     "body":"Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos. Object detection locates and classifies multiple objects in an image. Semantic segmentation assigns a class label to every pixel. Instance segmentation distinguishes individual object instances. Face recognition identifies or verifies individuals. Image generation uses generative adversarial networks and diffusion models to create synthetic images. Convolutional neural networks dominated computer vision until vision transformers emerged. OpenCV is a popular library for classical image processing operations."},
    {"url":"https://example.com/reinforcement-learning","title":"Reinforcement Learning",
     "domain":"example.com","snippet":"RL trains agents to make decisions by maximising cumulative reward.",
     "body":"Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximise the notion of cumulative reward. The agent learns without intervention from a human by maximising its reward and penalising bad decisions. Q-learning is a model-free reinforcement learning algorithm that learns the value of an action in a particular state. Policy gradient methods directly optimise the agent's policy. Actor-critic algorithms combine value-based and policy-based methods. Deep reinforcement learning uses neural networks to approximate value functions or policies. AlphaGo and AlphaZero demonstrated superhuman performance in board games using Monte Carlo tree search and deep RL."},
    {"url":"https://example.com/web-development","title":"Web Development: Frontend and Backend",
     "domain":"example.com","snippet":"Modern web development spans frontend interfaces and backend services.",
     "body":"Web development encompasses building and maintaining websites and web applications. Frontend development focuses on what users see and interact with using HTML, CSS, and JavaScript. React, Vue, and Angular are popular frontend frameworks. Backend development handles server logic, databases, and APIs using languages like Python, Node.js, Java, and Go. RESTful APIs and GraphQL enable communication between frontend and backend. HTTP and HTTPS are the protocols underlying web communication. Cookies, sessions, and JWT tokens manage user authentication. Web performance is measured by Core Web Vitals including Largest Contentful Paint, First Input Delay, and Cumulative Layout Shift."},
    {"url":"https://example.com/big-data","title":"Big Data and Apache Spark",
     "domain":"example.com","snippet":"Big data technologies process massive datasets across distributed clusters.",
     "body":"Big data refers to datasets that are too large or complex to be dealt with by traditional data processing application software. The characteristics of big data are often described using the three Vs: Volume, Velocity, and Variety. Apache Hadoop introduced the MapReduce programming model for distributed batch processing. Apache Spark improved upon Hadoop with in-memory processing, making it significantly faster for iterative algorithms. Apache Kafka enables real-time data streaming. Data lakes store raw data in native format until needed. ETL pipelines extract, transform, and load data between systems. Apache Parquet and ORC are columnar storage formats optimised for analytical queries."},
    {"url":"https://example.com/recommender","title":"Recommender Systems",
     "domain":"example.com","snippet":"Recommender systems predict user preferences to suggest relevant items.",
     "body":"A recommender system is a subclass of information filtering system that seeks to predict the rating or preference a user would give to an item. Collaborative filtering recommends items based on the preferences of similar users. Content-based filtering recommends items similar to those the user has liked in the past. Hybrid systems combine both approaches. Matrix factorisation decomposes the user-item interaction matrix into latent factor representations. Deep learning methods like neural collaborative filtering and autoencoders learn complex interaction patterns. Evaluation metrics include precision, recall, NDCG, mean average precision, and click-through rate."},
]

def seed_documents():
    """Insert seed documents into DB if empty."""
    con = get_con()
    count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if count == 0:
        print("[Explorer] Seeding initial corpus…")
        for doc in SEED_DOCS:
            con.execute("""
                INSERT OR IGNORE INTO documents
                (url, title, body, snippet, domain, crawled_at, doc_hash)
                VALUES (?,?,?,?,?,?,?)
            """, (doc["url"], doc["title"], doc["body"], doc["snippet"],
                  doc["domain"], datetime.utcnow().isoformat(),
                  hashlib.md5(doc["body"].encode()).hexdigest()))
        con.commit()
        print(f"[Explorer] {len(SEED_DOCS)} seed documents inserted.")
    con.close()


# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query:   str
    page:    int = 1
    top_k:   int = RESULTS_N

class CrawlRequest(BaseModel):
    urls:      List[str]
    max_pages: int = 20

class IndexDocRequest(BaseModel):
    url:     str
    title:   str
    body:    str
    snippet: str = ""
    domain:  str = ""

class WikipediaIndexRequest(BaseModel):
    topics:    List[str]
    max_chars: int = 5000   # chars of body to keep per article


# ─────────────────────────────────────────────────────────────
# WIKIPEDIA API INDEXER
# ─────────────────────────────────────────────────────────────
def fetch_wikipedia_article(title: str, max_chars: int = 5000) -> Optional[dict]:
    """Fetch a Wikipedia article via the official API (no blocking)."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action":         "query",
        "titles":         title,
        "prop":           "extracts",
        "explaintext":    True,
        "exsectionformat":"plain",
        "format":         "json",
        "redirects":      1,
    }
    try:
        r = requests.get(url, params=params,
                         headers={"User-Agent": "TheExplorer/1.0 (educational)"},
                         timeout=10)
        pages = r.json()["query"]["pages"]
        page  = next(iter(pages.values()))
        if "extract" not in page or not page["extract"].strip():
            return None
        body    = page["extract"][:max_chars]
        snippet = body[:200] + "…"
        wiki_url = f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"
        return {
            "url":     wiki_url,
            "title":   page["title"],
            "body":    body,
            "snippet": snippet,
            "domain":  "en.wikipedia.org",
        }
    except Exception as e:
        print(f"[Wiki] Error fetching '{title}': {e}")
        return None


def run_wikipedia_index_job(topics: List[str], max_chars: int = 5000):
    """Background job: fetch and index a list of Wikipedia topics."""
    con = get_con()
    success = 0
    for i, topic in enumerate(topics):
        doc = fetch_wikipedia_article(topic, max_chars)
        if doc:
            try:
                con.execute("""
                    INSERT OR REPLACE INTO documents
                    (url, title, body, snippet, domain, crawled_at, doc_hash)
                    VALUES (?,?,?,?,?,?,?)
                """, (doc["url"], doc["title"], doc["body"], doc["snippet"],
                      doc["domain"], datetime.utcnow().isoformat(),
                      hashlib.md5(doc["body"].encode()).hexdigest()))
                con.commit()
                success += 1
                print(f"[Wiki] [{i+1}/{len(topics)}] ✓ {doc['title']}")
            except Exception as e:
                print(f"[Wiki] DB error for '{topic}': {e}")
        time.sleep(0.3)   # polite delay — Wikipedia API is free but be respectful
    con.close()
    rebuild_index()
    print(f"[Wiki] Done. {success}/{len(topics)} articles indexed.")


# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    seed_documents()
    rebuild_index()

@app.get("/")
def root():
    return {"name": "The Explorer Search API", "version": "2.0.0",
            "docs": "/docs", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok", "docs_indexed": len(doc_store),
            "model": MODEL_NAME, "faiss": USE_FAISS,
            "wordnet": WORDNET_AVAILABLE}

@app.get("/search")
def search(q: str, page: int = 1, top_k: int = RESULTS_N):
    """Main search endpoint — GET /search?q=cat&page=1"""
    if not q.strip():
        raise HTTPException(400, "Query cannot be empty")
    return hybrid_search(q.strip(), top_k=top_k, page=page)

@app.post("/search")
def search_post(req: SearchRequest):
    """POST /search — body: {query, page, top_k}"""
    return hybrid_search(req.query.strip(), top_k=req.top_k, page=req.page)

@app.post("/index")
def index_document(req: IndexDocRequest):
    """Manually add a single document to the index."""
    con = get_con()
    con.execute("""
        INSERT OR REPLACE INTO documents
        (url, title, body, snippet, domain, crawled_at, doc_hash)
        VALUES (?,?,?,?,?,?,?)
    """, (req.url, req.title, req.body, req.snippet,
          req.domain or urllib.parse.urlparse(req.url).netloc,
          datetime.utcnow().isoformat(),
          hashlib.md5(req.body.encode()).hexdigest()))
    con.commit()
    con.close()
    rebuild_index()
    return {"status": "indexed", "url": req.url}

@app.post("/index/wikipedia")
def index_wikipedia(req: WikipediaIndexRequest, background_tasks: BackgroundTasks):
    """
    Index a list of Wikipedia topics via the official Wikipedia API.
    POST /index/wikipedia
    Body: {"topics": ["Cat", "Aeroplane", "House", ...], "max_chars": 5000}
    """
    background_tasks.add_task(run_wikipedia_index_job, req.topics, req.max_chars)
    return {
        "status":  "wikipedia indexing started",
        "topics":  len(req.topics),
        "message": "Articles are being fetched in the background. Check /stats for progress."
    }

@app.post("/crawl")
async def crawl(req: CrawlRequest, background_tasks: BackgroundTasks):
    """Start a background crawl job from seed URLs (for non-Wikipedia sites)."""
    background_tasks.add_task(run_crawl_job, req.urls, req.max_pages)
    return {"status": "crawl started", "seeds": req.urls, "max_pages": req.max_pages}

@app.get("/stats")
def stats():
    """Return engine statistics."""
    con = get_con()
    doc_count   = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    query_count = con.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]
    top_queries = con.execute(
        "SELECT query, COUNT(*) as n FROM search_log GROUP BY query ORDER BY n DESC LIMIT 10"
    ).fetchall()
    avg_latency = con.execute("SELECT AVG(latency) FROM search_log").fetchone()[0]
    con.close()
    return {
        "docs_indexed":    doc_count,
        "documents":       doc_count,
        "queries_served":  query_count,
        "avg_latency_ms":  round(avg_latency or 0, 2),
        "top_queries":     [{"query": r[0], "count": r[1]} for r in top_queries],
        "model":           MODEL_NAME,
        "embed_dim":       EMBED_DIM,
        "wordnet_enabled": WORDNET_AVAILABLE,
    }

@app.get("/suggest")
def suggest(q: str, limit: int = 6):
    """Autocomplete suggestions from search history."""
    if not q or len(q) < 2:
        return {"suggestions": []}
    con = get_con()
    rows = con.execute(
        "SELECT DISTINCT query FROM search_log WHERE query LIKE ? ORDER BY rowid DESC LIMIT ?",
        (f"{q}%", limit)
    ).fetchall()
    con.close()
    return {"suggestions": [r[0] for r in rows]}

@app.delete("/index")
def clear_index():
    """Clear all indexed documents."""
    con = get_con()
    con.execute("DELETE FROM documents")
    con.commit()
    con.close()
    rebuild_index()
    return {"status": "index cleared"}


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
