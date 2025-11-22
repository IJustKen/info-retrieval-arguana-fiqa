import os
import json
import pickle
import re
from nltk.stem import PorterStemmer
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import threading
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def load_beir_dataset(dataset_name: str):
    """Load BEIR dataset with proper qrels handling"""
    import ir_datasets

    dataset_map = {
        'nq': 'beir/nq',
        'arguana': 'beir/arguana',
        'treccovid': 'beir/trec-covid',
        'fiqa': 'beir/fiqa/test',  # Important: use /test split
    }

    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset must be one of {list(dataset_map.keys())}, got {dataset_name}")

    dataset_path = dataset_map[dataset_name]
    print(f"[DATASET] Loading {dataset_name} from {dataset_path}...")
    dataset = ir_datasets.load(dataset_path)

    corpus = {}
    queries = {}
    qrels = defaultdict(dict)

    # Load documents
    print(f"[DATASET] Loading documents...")
    doc_count = 0
    for doc in dataset.docs_iter():
        corpus[doc.doc_id] = {
            'title': doc.title if hasattr(doc, 'title') else '',
            'text': doc.text if hasattr(doc, 'text') else '',
        }
        doc_count += 1
        if doc_count % 10000 == 0:
            print(f"[DATASET] Loaded {doc_count} documents...")
    print(f"[DATASET] ✓ Loaded {len(corpus)} documents")
    
    # Load queries
    print(f"[DATASET] Loading queries...")
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
    print(f"[DATASET] ✓ Loaded {len(queries)} queries")

    # Load qrels
    print(f"[DATASET] Loading qrels...")
    qrels_loaded = False
    qrels_count = 0
    
    # Check if dataset has qrels_iter 
    print(f"[DATASET] Checking for qrels_iter...")
    print(f"[DATASET] Has qrels_iter: {hasattr(dataset, 'qrels_iter')}")
    print(f"[DATASET] Has qrels_dict: {hasattr(dataset, 'qrels_dict')}")
    
    # Try qrels_iter() - this usually works for all BEIR datasets
    try:
        print(f"[DATASET] Attempting qrels_iter()...")
        for qrel in dataset.qrels_iter():
            # qrel has: query_id, doc_id, relevance, iteration
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            qrels_count += 1
            if qrels_count % 1000 == 0:
                print(f"[DATASET] Loaded {qrels_count} qrels...")
        qrels_loaded = True
        print(f"[DATASET] ✓ Loaded {qrels_count} qrels using qrels_iter()")
    except AttributeError as e:
        print(f"[DATASET] ✗ qrels_iter() AttributeError: {e}")
    except Exception as e:
        print(f"[DATASET] ✗ qrels_iter() Exception: {e}")
        import traceback
        traceback.print_exc()
    
    if not qrels_loaded:
        print(f"[DATASET] Warning: No qrels loaded")
        print(f"[DATASET] Dataset info:")
        print(f"[DATASET]   - dataset_id: {dataset.dataset_id if hasattr(dataset, 'dataset_id') else 'N/A'}")
        print(f"[DATASET]   - docs_count: {dataset.docs_count() if hasattr(dataset, 'docs_count') else 'N/A'}")
        print(f"[DATASET]   - queries_count: {dataset.queries_count() if hasattr(dataset, 'queries_count') else 'N/A'}")

    print(f"[DATASET] Dataset loading complete:")
    print(f"[DATASET]   - Corpus: {len(corpus)} documents")
    print(f"[DATASET]   - Queries: {len(queries)} queries")
    print(f"[DATASET]   - Qrels: {len(qrels)} query-document pairs")

    return corpus, queries, dict(qrels)

# Evaluation metric functions

def compute_ndcg(rankings: List[str], relevant_docs: Dict[str, int], k: int = 10) -> float:
    """Compute NDCG@k"""
    dcg = 0.0
    idcg = 0.0

    for i, doc_id in enumerate(rankings[:k]):
        relevance = relevant_docs.get(doc_id, 0)
        dcg += relevance / np.log2(i + 2)

    ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
    for i, rel in enumerate(ideal_relevances):
        idcg += rel / np.log2(i + 2)

    return (dcg / idcg) if idcg > 0 else 0.0

def compute_precision_at_k(rankings: List[str], relevant_docs: Dict[str, int], k: int = 10) -> float:
    """Compute Precision@k"""
    relevant_count = sum(1 for doc_id in rankings[:k] if doc_id in relevant_docs)
    return relevant_count / k if k > 0 else 0.0

def compute_map_at_k(rankings: List[str], relevant_docs: Dict[str, int], k: int = 10) -> float:
    """Compute MAP@k"""
    precisions = []
    relevant_count = 0

    for i, doc_id in enumerate(rankings[:k]):
        if doc_id in relevant_docs:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))

    return (sum(precisions) / k) if k > 0 else 0.0


# RETRIEVERS

class BM25Retriever:
    def __init__(self):
        self.inverted_index = {}  # term -> list of (doc_id, tf)
        self.doc_lengths = {}     # doc_id -> length
        self.idf = {}             # term -> idf score
        self.avgdl = 0
        self.N = 0                # Total docs
        self.k1 = 1.5             # initialize params for BM25 k1 and b
        self.b = 0.75


        self.stemmer = PorterStemmer()
        # Pre-compile regex for speed (removes punctuation)
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s]')

    def _preprocess(self, text: str) -> list:
        """
        1. Lowercase
        2. Remove punctuation
        3. Tokenize (split)
        4. Stemming
        """
        # Lowercase & Remove Punctuation (keep only letters/numbers)
        text = self.clean_pattern.sub(' ', text.lower())
        
        # Tokenize
        tokens = text.split()
        
        # Stemming
        stemmed_tokens = [self.stemmer.stem(t) for t in tokens]
        
        return stemmed_tokens
        
    def train(self, corpus: Dict):
        """Build inverted index for BM25"""
        import time
        import numpy as np
        from collections import defaultdict
        
        start = time.time()
        print("[BM25] Building inverted index...")
        
        self.N = len(corpus)
        total_doc_length = 0
        
        # Temporary index: term -> list of (doc_id, tf)
        inverted_index_temp = defaultdict(list)
        
        for i, (doc_id, doc_data) in enumerate(corpus.items()):
            if i % 10000 == 0:
                print(f"[BM25] Indexing: {i}/{self.N} ({100*i/self.N:.1f}%)")

            raw_text = (doc_data.get('title', '') + ' ' + doc_data.get('text', '')).strip()

            tokens = self._preprocess(raw_text)
            doc_length = len(tokens)
            
            self.doc_lengths[doc_id] = doc_length
            total_doc_length += doc_length
            
            # Count frequencies for this document
            # This is fast for short documents
            term_counts = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            
            # Add to index
            for term, tf in term_counts.items():
                inverted_index_temp[term].append((doc_id, tf))
                
        self.inverted_index = dict(inverted_index_temp)
        self.avgdl = total_doc_length / self.N if self.N > 0 else 0
        
        print(f"[BM25] Calculating IDF scores...")
        for term, postings in self.inverted_index.items():
            n_q = len(postings)  # Number of docs containing term
            # Standard BM25 formula
            self.idf[term] = np.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            
        elapsed = time.time() - start
        print(f"[BM25] ✓ Indexed {self.N} documents in {elapsed:.1f} seconds")
        print(f"[BM25] Vocabulary size: {len(self.inverted_index)} terms")

    def retrieve(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """
        Retrieve using Accumulator Pattern
        Instead of finding candidates -> scoring each, we score terms -> updating candidates.
        """
        import time
        import heapq
        from collections import defaultdict
        
        start = time.time()
        
        # Tokenize Query
        query_tokens = self._preprocess(query)
        
        # Accumulator (The Scoreboard)
        # Keys are DocIDs, Values are partial scores
        scores = defaultdict(float)
        
        # Iterate Term-at-a-Time
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            
            # Pre-fetch global stats for this term
            idf = self.idf[token]
            postings = self.inverted_index[token] # List of (doc_id, tf)
            
            # Iterate through the pre-made list of docs for this term
            # We ONLY touch docs that match this specific term
            for doc_id, tf in postings:
                doc_len = self.doc_lengths[doc_id]
                
                # BM25 term saturation logic
                # This part is fast math operations
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                term_score = idf * (numerator / denominator)
                
                # Update the scoreboard
                scores[doc_id] += term_score

        # Efficient Top-K Selection
        # If we have huge number of scores, heapq is O(NlogK) which is faster than sorted() O(N log N)
        top_k_results = heapq.nlargest(k, scores.items(), key=lambda x: x[1])
        
        elapsed = time.time() - start
        return top_k_results




class TFIDFRetriever:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000,  # Limit vocabulary size
            min_df=2,  # Ignore terms that appear in < 2 documents
            max_df=0.8  # Ignore terms that appear in > 80% of documents
        )
        self.stemmer = PorterStemmer()
        # Pre-compile regex for speed (removes punctuation)
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        self.doc_vectors = None
        self.doc_ids = None
        self.corpus_dict = None
    def _preprocess(self, text: str) -> list:
        """
        1. Lowercase
        2. Remove punctuation
        3. Tokenize (split)
        4. Stemming
        """
        # Lowercase & Remove Punctuation (keep only letters/numbers)
        text = self.clean_pattern.sub(' ', text.lower())
        
        # Tokenize
        tokens = text.split()
        
        # Stemming
        stemmed_tokens = [self.stemmer.stem(t) for t in tokens]
        
        return stemmed_tokens
    
    def train(self, corpus: Dict):
        """Train TF-IDF on corpus"""
        import time
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        start = time.time()
        print("[TF-IDF] Building TF-IDF index...")
        
        self.corpus_dict = corpus
        corpus_texts = []
        doc_ids = []
        
        for doc_id, doc_data in corpus.items():
            raw_text = (doc_data.get('title', '') + ' ' + doc_data.get('text', '')).strip()
        
            proc_tokens = self._preprocess(raw_text)
            proc_string = " ".join(proc_tokens)       # Join back to string "run fast"
            corpus_texts.append(proc_string)
            doc_ids.append(doc_id)
        
        print(f"[TF-IDF] Vectorizing {len(corpus_texts)} documents...")
        
        # Fit and transform corpus
        self.doc_vectors = self.vectorizer.fit_transform(corpus_texts)
        self.doc_ids = doc_ids
        
        elapsed = time.time() - start
        print(f"[TF-IDF] Indexed {len(corpus)} documents in {elapsed:.1f} seconds")
        print(f"[TF-IDF] Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def retrieve(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top-k documents using TF-IDF"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if self.doc_vectors is None:
            raise ValueError("TF-IDF not trained")
        
        # Transform query using same vocabulary
        query_tokens = self._preprocess(query)    # Returns list
        query_string = " ".join(query_tokens)     # Join back to string
        query_vector = self.vectorizer.transform([query_string])
        # Compute cosine similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return (doc_id, score) tuples
        retrieved = []
        for idx in top_k_indices:
            doc_id = self.doc_ids[idx]
            score = float(similarities[idx])
            retrieved.append((doc_id, score))
        
        return retrieved


class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        import time
        start = time.time()
        
        print(f"[CrossEncoder] Loading model: {model_name}")
        print(f"[CrossEncoder] Started (downloading model)...")
        print(f"[CrossEncoder] Subsequent loads will be instant (cached)")
        
        try:
            self.model = CrossEncoder(model_name, max_length=512)
            elapsed = time.time() - start
            print(f"[CrossEncoder] Model loaded successfully in {elapsed:.1f} seconds")
        except Exception as e:
            print(f"[CrossEncoder] Error loading model: {e}")
            raise


    def rerank(self, query: str, candidates: List[Tuple[str, float]], corpus: Dict) -> List[Tuple[str, float]]:
        """Rerank using cross-encoder"""
        if not candidates:
            return candidates

        pairs = []
        doc_ids = []

        for doc_id, _ in candidates:
            if doc_id not in corpus:
                continue

            doc = corpus[doc_id]
            doc_text = (doc.get('title', '') + ' ' + doc.get('text', '')).strip()
            pairs.append((query, doc_text))
            doc_ids.append(doc_id)

        if not pairs:
            return candidates

        scores = self.model.predict(pairs)
        reranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        return reranked

# FLASK APP
app = Flask(__name__)
CORS(app)

# Global state
pipelines = {}  # Store trained models per dataset
training_status = {}  # Track training progress

class Pipeline:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.lower()
        self.corpus = None
        self.queries = None
        self.qrels = None
        self.bm25_retriever = None
        self.tfidf_retriever = None
        self.cross_encoder_reranker = None
        self.is_trained = False
        self.spell_checker = None

    def load_data(self):
        """Load dataset"""
        self.corpus, self.queries, self.qrels = load_beir_dataset(self.dataset_name)

    def train(self):
        """Train all models"""
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.train(self.corpus)
        

        self.tfidf_retriever = TFIDFRetriever()
        self.tfidf_retriever.train(self.corpus)

        self.cross_encoder_reranker = CrossEncoderReranker()
        print("[Pipeline] Initializing Spell Checker...")
        vocab_list = list(self.bm25_retriever.inverted_index.keys()) 
        self.spell_checker = SpellChecker(vocab_list)

        self.is_trained = True

    def retrieve(self, query: str, method: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve using specified method"""
        if not self.is_trained:
            raise ValueError("Pipeline not trained")

        if method == 'bm25':
            candidates = self.bm25_retriever.retrieve(query, k=k)
            return candidates

        elif method == 'tfidf':
            candidates = self.tfidf_retriever.retrieve(query, k=k)
            return candidates

        elif method == 'crossencoder':
            # Cross-encoder needs BM25 candidates first
            candidates = self.bm25_retriever.retrieve(query, k=k*10)
            reranked = self.cross_encoder_reranker.rerank(query, candidates, self.corpus)
            return reranked[:k]

        else:
            # Default to BM25
            candidates = self.bm25_retriever.retrieve(query, k=k)
            return candidates

    def evaluate(self, num_queries: int = 50):
        """Evaluate all methods"""
        query_ids = list(self.qrels.keys())[:num_queries]
        results = {
            'bm25': defaultdict(list),
            'tfidf': defaultdict(list),
            'crossencoder': defaultdict(list),
        }

        counts = 1
        for query_id in query_ids[:25]:
            print(f"Query {query_id} is being queried - {counts}/{len(query_ids)}")
            counts += 1
            query_text = self.queries[query_id]
            relevant_docs = self.qrels[query_id]

            try:
                bm25_rankings = self.retrieve(query_text, 'bm25', k=10)
                bm25_doc_ids = [doc_id for doc_id, _ in bm25_rankings]

                results['bm25']['ndcg@10'].append(compute_ndcg(bm25_doc_ids, relevant_docs, k=10))
                results['bm25']['p@5'].append(compute_precision_at_k(bm25_doc_ids, relevant_docs, k=5))
                results['bm25']['p@10'].append(compute_precision_at_k(bm25_doc_ids, relevant_docs, k=10))
                results['bm25']['map@10'].append(compute_map_at_k(bm25_doc_ids, relevant_docs, k=10))
            except:
                pass

            try:
                lm_rankings = self.retrieve(query_text, 'tfidf', k=10)
                lm_doc_ids = [doc_id for doc_id, _ in lm_rankings]

                results['tfidf']['ndcg@10'].append(compute_ndcg(lm_doc_ids, relevant_docs, k=10))
                results['tfidf']['p@5'].append(compute_precision_at_k(lm_doc_ids, relevant_docs, k=5))
                results['tfidf']['p@10'].append(compute_precision_at_k(lm_doc_ids, relevant_docs, k=10))
                results['tfidf']['map@10'].append(compute_map_at_k(lm_doc_ids, relevant_docs, k=10))
            except:
                pass

            try:
                ce_rankings = self.retrieve(query_text, 'crossencoder', k=10)
                ce_doc_ids = [doc_id for doc_id, _ in ce_rankings]

                results['crossencoder']['ndcg@10'].append(compute_ndcg(ce_doc_ids, relevant_docs, k=10))
                results['crossencoder']['p@5'].append(compute_precision_at_k(ce_doc_ids, relevant_docs, k=5))
                results['crossencoder']['p@10'].append(compute_precision_at_k(ce_doc_ids, relevant_docs, k=10))
                results['crossencoder']['map@10'].append(compute_map_at_k(ce_doc_ids, relevant_docs, k=10))
            except:
                pass

        summary = {}
        for model_name, metrics in results.items():
            summary[model_name] = {}
            for metric_name, values in metrics.items():
                summary[model_name][metric_name] = float(np.mean(values)) if values else 0.0

        return summary
    
import requests
import time

from groq import Groq

class RAGGenerator:
    def __init__(self, groq_api_key):
        self.client = Groq(api_key=groq_api_key)
        print("[RAG] Initialized Groq API Agent")
        print("[RAG] Using model: mixtral-8x7b-32768 (fast, free)")

    def generate_overview(self, query, documents):
        """Generate true RAG summary using Groq LLM"""
        if not documents:
            return "No documents found to generate answer."

        # Prepare Context from top 5 docs
        context = ""
        for i, doc in enumerate(documents[:5], 1):
            title = doc.get('title', 'Untitled')
            text = doc.get('text', '')[:500]  # Limit per doc
            context += f"Source {i} ({title}):\n{text}\n\n"

        # Create prompt
        prompt = f"""Based on the following sources, answer the user's question concisely in 3-4 sentences.



Question: {query}

Sources:
{context}

Answer:"""

        try:
            print("[RAG] Calling Groq API...")
            # Call Groq
            message = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant", 
                max_tokens=150,
                temperature=0.3  # Low creativity = more factual
            )
            
            answer = message.choices[0].message.content.strip()
            print(f"[RAG] ✅ Generated summary: {answer[:100]}...")
            return answer
            
        except Exception as e:
            print(f"[RAG] ❌ Error: {e}")
            return "AI Overview unavailable."
        
class SpellChecker:
    def __init__(self, vocabulary):
        self.vocab = set(vocabulary)
        print(f"[SpellChecker] Initialized with {len(self.vocab)} words")

    def levenshtein(self, s1, s2):
        """
        Standard Levenshtein Distance Algorithm (Iterative)
        Returns the minimum number of edits (insert, delete, substitute)
        """
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        # Initialize matrix rows
        # We only need two rows (previous and current) to save memory
        # This is a standard optimization taught in DSA
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost is 0 if characters match, else 1
                cost = 0 if c1 == c2 else 1
                
                # Calculate minimum of:
                # Deletion (current_row[j] + 1)
                # Insertion (previous_row[j+1] + 1)
                # Substitution (previous_row[j] + cost)
                current_row.append(min(
                    previous_row[j+1] + 1,
                    current_row[j] + 1,
                    previous_row[j] + cost
                ))
            previous_row = current_row
            
        return previous_row[-1]
    
    def correct_query(self, query):
        """Corrects words in query if they are close to vocab words"""
        tokens = query.lower().split()
        corrected_tokens = []
        
        for token in tokens:
            if token in self.vocab:
                corrected_tokens.append(token)
                continue
                
            # Simple optimization: check only words with similar length
            candidates = [w for w in self.vocab if abs(len(w) - len(token)) <= 2]
            
            if candidates:
                best_word = min(candidates, key=lambda w: self.levenshtein(token, w))
                if self.levenshtein(token, best_word) <= 2: # Threshold
                    corrected_tokens.append(best_word)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
                
        return " ".join(corrected_tokens)


groq_api_key = "API_KEY"  # Get from https://console.groq.com/keys
rag_agent = RAGGenerator(groq_api_key)


# API ENDPOINTS

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get available datasets"""
    return jsonify({
        'datasets': [
            {'name': 'Natural Questions', 'id': 'nq', 'corpus_size': 2681468, 'num_queries': 3452},
            {'name': 'ArguAna', 'id': 'arguana', 'corpus_size': 8674, 'num_queries': 1406},
           {'name': 'FiQA-2018', 'id': 'fiqa', 'corpus_size': 57638, 'num_queries': 648}
            ]
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train models for a dataset"""
    data = request.json
    dataset_name = data.get('dataset', 'nq')

    if dataset_name in training_status and training_status[dataset_name]['status'] == 'training':
        return jsonify({'status': 'already_training'}), 400

    def train_in_background():
        try:
            training_status[dataset_name] = {'status': 'training', 'progress': 0}

            pipeline = Pipeline(dataset_name)
            training_status[dataset_name]['progress'] = 10

            pipeline.load_data()
            training_status[dataset_name]['progress'] = 20

            print(f"[TRAINING] Data loaded. Starting model training..")

            pipeline.train()
            training_status[dataset_name]['progress'] = 100
            training_status[dataset_name]['status'] = 'completed'

            print(f"[TRAINING] Finished training for {dataset_name}")

            pipelines[dataset_name] = pipeline
        except Exception as e:
            print(f"[TRAINING] Error training {dataset_name}: {str(e)}")
            training_status[dataset_name] = {'status': 'error', 'error': str(e)}
            import traceback
            traceback.print_exc() 
            training_status[dataset_name] = {'status': 'error', 'error': str(e)}

    thread = threading.Thread(target=train_in_background)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'training_started'})

@app.route('/api/training-status/<dataset_name>', methods=['GET'])
def get_training_status(dataset_name):
    """Get training status"""
    status = training_status.get(dataset_name, {'status': 'not_started'})
    is_trained = dataset_name in pipelines and pipelines[dataset_name].is_trained

    return jsonify({
        **status,
        'is_trained': is_trained
    })

import time
@app.route('/api/search', methods=['POST'])
def search():
    start_total = time.time()
    """Search using specified method"""
    
    data = request.json
    dataset_name = data.get('dataset', 'nq')
    query = data.get('query', '')
    method = data.get('method', 'bm25')

    if dataset_name not in pipelines or not pipelines[dataset_name].is_trained:
        return jsonify({'error': 'Pipeline not trained'}), 400

    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    pipeline = pipelines[dataset_name]

    try:
        corrected_query = query
        # Check if spell checker exists and is trained
        if pipeline.spell_checker:
            corrected_query = pipeline.spell_checker.correct_query(query)
        did_you_mean = corrected_query if corrected_query != query else None
        search_start = time.time()
        results = pipeline.retrieve(corrected_query, method, k=10)
        search_time = (time.time()-search_start) * 1000
        
        # Prepare results for frontend
        output_results = []
        full_docs_for_rag = []  # Store full docs for RAG
        
        for rank, (doc_id, score) in enumerate(results, 1):
            doc = pipeline.corpus.get(doc_id, {})
            title = doc.get('title', 'N/A')[:100]
            text = doc.get('text', '')[:200]

            # Generate simple summary
            text_lines = text.split('. ')
            summary = '. '.join(text_lines[:2]) + '.' if text_lines else 'N/A'

            output_results.append({
                'rank': rank,
                'doc_id': doc_id,
                'score': float(score),
                'title': title,
                'summary': summary
            })
            
            # Store full document for RAG (not just snippet)
            full_docs_for_rag.append({
                'title': doc.get('title', ''),
                'text': doc.get('text', '')
            })

        # Generate AI Overview
        print(f"[Search] Generating AI Overview for query: {query}")
        rag_start = time.time()
        ai_overview = rag_agent.generate_overview(corrected_query, full_docs_for_rag)
        rag_time = (time.time()-rag_start)*1000
        total_time = (time.time()-start_total)*1000
        
        return jsonify({
            'results': output_results,
            'ai_overview': ai_overview,
            'did_you_mean': did_you_mean,
            'metrics':{'search_time_ms': round(search_time, 2),
            'rag_time_ms': round(rag_time, 2),
            'total_time_ms': round(total_time, 2)
        }
    })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate all methods"""
    data = request.json
    dataset_name = data.get('dataset', 'nq')

    if dataset_name not in pipelines or not pipelines[dataset_name].is_trained:
        return jsonify({'error': 'Pipeline not trained'}), 400

    try:
        pipeline = pipelines[dataset_name]
        summary = pipeline.evaluate(num_queries=25)

        # Calculate improvements
        improvements = {}
        for method in ['crossencoder']:
            improvements[method] = {}
            for metric in ['ndcg@10', 'p@5', 'p@10', 'map@10']:
                bm25_val = summary['bm25'][metric]
                method_val = summary[method][metric]
                if bm25_val > 0:
                    imp = ((method_val - bm25_val) / bm25_val) * 100
                    improvements[method][metric] = float(imp)

        return jsonify({
            'summary': summary,
            'improvements': improvements
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting IR Comparison System Backend...")
    print("API available at http://localhost:5000")
    app.run(debug=True, threaded=True, port=5000)
