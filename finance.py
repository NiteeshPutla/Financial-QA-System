import os
import json
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from bs4 import BeautifulSoup
import google.generativeai as genai
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv
from math import log
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a 10-K filing"""
    text: str
    company: str
    year: str
    page: int = 0
    section: str = ""
    
@dataclass
class QueryResult:
    """Structured query result"""
    query: str
    answer: str
    reasoning: str
    sub_queries: List[str]
    sources: List[Dict[str, Any]]

class SECFilingDownloader:
    """Downloads 10-K filings from SEC EDGAR database"""
    
    BASE_URL = "https://www.sec.gov/Archives/edgar/data"
    SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    COMPANY_CIKS = {
        "GOOGL": "1652044",
        "MSFT": "789019", 
        "NVDA": "1045810"
    }
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        # Required headers for SEC
        self.session.headers.update({
            'User-Agent': 'Financial RAG System financial.rag@example.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
    
    def download_filing(self, company: str, year: str) -> Optional[str]:
        """Download 10-K filing for company and year"""
        filename = f"{company}_{year}_10K.html"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"Using cached filing: {filename}")
            return str(filepath)
        
        logger.info(f"Downloading {company} {year} 10-K filing...")
        
        try:
            # Get filing list
            cik = self.COMPANY_CIKS[company]
            search_params = {
                'CIK': cik,
                'type': '10-K',
                'dateb': f"{int(year) + 1}1231",
                'count': '10'
            }
            
            search_response = self.session.get(self.SEARCH_URL, params=search_params)
            search_response.raise_for_status()
            
            # Parse search results to find filing URL
            # This is simplified - in practice you'd parse the HTML more robustly
            html = search_response.text
            
            # Look for 10-K filing link (simplified regex)
            pattern = r'href="(/Archives/edgar/data/\d+/\d+-\d+-\d+\.htm)"'
            matches = re.findall(pattern, html)
            
            if not matches:
                logger.error(f"No 10-K filing found for {company} {year}")
                return None
            
            # Get the first match (most recent)
            filing_path = matches[0]
            filing_url = f"https://www.sec.gov{filing_path}"
            
            # Download the filing
            filing_response = self.session.get(filing_url)
            filing_response.raise_for_status()
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(filing_response.text)
            
            logger.info(f"Downloaded: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading {company} {year}: {e}")
            return None
        
    def download_all_filings(self) -> Dict[str, Dict[str, str]]:
        """Download all required filings"""
        filings = {}
        
        for company in ["GOOGL", "MSFT", "NVDA"]:
            filings[company] = {}
            for year in ["2022", "2023", "2024"]:
                filepath = self.download_filing(company, year)
                if filepath:
                    filings[company][year] = filepath
        
        return filings

class TextProcessor:
    """Processes and chunks 10-K filing text"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_html(self, filepath: str) -> str:
        """Extract clean text from HTML filing"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'lxml')
            # Remove scripts/styles
            for tag in soup(['script', 'style']):
                tag.decompose()
            text = soup.get_text(separator=' ')
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {filepath}: {e}")
            return ""

    def extract_tables_from_html(self, filepath: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse HTML tables into JSON rows; returns (rows, table_titles)."""
        rows: List[Dict[str, Any]] = []
        titles: List[str] = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, 'lxml')
            tables = soup.find_all('table')
            for t_idx, table in enumerate(tables):
                # Try to find a title/caption nearby
                title = None
                if table.caption and table.caption.get_text(strip=True):
                    title = table.caption.get_text(strip=True)
                else:
                    # Look at preceding sibling text for a likely title
                    prev = table.find_previous(string=True)
                    if prev:
                        cand = prev.strip()
                        if 3 <= len(cand) <= 200:
                            title = cand
                title = title or f"Table {t_idx+1}"
                titles.append(title)
                # Extract headers
                headers: List[str] = []
                header_row = table.find('tr')
                if header_row:
                    ths = header_row.find_all(['th'])
                    if ths:
                        headers = [th.get_text(strip=True) or f"col_{i}" for i, th in enumerate(ths)]
                if not headers:
                    # Try first row tds as headers
                    first = table.find('tr')
                    if first:
                        tds = first.find_all('td')
                        headers = [td.get_text(strip=True) or f"col_{i}" for i, td in enumerate(tds)]
                # Extract data rows
                trs = table.find_all('tr')
                for r_idx, tr in enumerate(trs[1:] if len(trs) > 1 else trs):
                    cells = tr.find_all(['td', 'th'])
                    values = [c.get_text(separator=' ', strip=True) for c in cells]
                    if not values:
                        continue
                    # Align with headers
                    row_dict: Dict[str, Any] = {}
                    for i, val in enumerate(values):
                        key = headers[i] if i < len(headers) else f"col_{i}"
                        row_dict[key] = val
                    row_dict['__table_title'] = title
                    row_dict['__row_index'] = r_idx
                    rows.append(row_dict)
        except Exception as e:
            logger.error(f"Error extracting tables from {filepath}: {e}")
        return rows, titles
    
    def chunk_text(self, text: str, company: str, year: str) -> List[DocumentChunk]:
        """Split text into semantic chunks"""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Estimate page number (rough approximation)
            page = i // 200 + 1
            
            chunk = DocumentChunk(
                text=chunk_text,
                company=company,
                year=year,
                page=page
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_filing(self, filepath: str, company: str, year: str) -> List[DocumentChunk]:
        """Process a single filing into text chunks and table-row chunks"""
        chunks: List[DocumentChunk] = []
        text = self.extract_text_from_html(filepath)
        chunks.extend(self.chunk_text(text, company, year))
        # Tables → JSON → chunks
        table_rows, _ = self.extract_tables_from_html(filepath)
        if table_rows:
            # Persist table JSON for transparency
            try:
                out_dir = Path(filepath).parent
                out_path = out_dir / f"{company}_{year}_tables.json"
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(table_rows, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not write tables JSON: {e}")
            for row in table_rows:
                table_title = row.get('__table_title', '')
                row_index = int(row.get('__row_index', 0))
                # Build a readable text representation of the row
                kv_pairs = []
                for k, v in row.items():
                    if k.startswith('__'):
                        continue
                    if v:
                        kv_pairs.append(f"{k}: {v}")
                row_text = f"TABLE {table_title} | " + " | ".join(kv_pairs)
                chunks.append(DocumentChunk(
                    text=row_text,
                    company=company,
                    year=year,
                    page=0,
                    section=f"table:{table_title}#row{row_index}"
                ))
        return chunks

class VectorStore:
    """In-memory vector store using OpenAI embeddings + keyword fallback."""
    
    def __init__(self, openai_client: Optional["OpenAI"] = None, embedding_model: str = "text-embedding-3-large"):
        self.chunks: List[DocumentChunk] = []
        self._openai = openai_client
        self._embedding_model = embedding_model
        self._embeddings: Optional[np.ndarray] = None  # shape: (N, D)
        # Keyword index
        self._doc_term_freqs: List[Dict[str, int]] = []
        self._doc_lengths: List[int] = []
        self._df: Dict[str, int] = {}
        self._vocab: set = set()
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")
    
    def _compute_keyword_index(self, texts: List[str]):
        self._doc_term_freqs.clear()
        self._doc_lengths.clear()
        self._df.clear()
        self._vocab = set()
        for text in texts:
            # Simple tokenization: alphanumerics lowered
            tokens = re.findall(r"[a-zA-Z0-9$%]+", text.lower())
            term_freq: Dict[str, int] = {}
            for tok in tokens:
                term_freq[tok] = term_freq.get(tok, 0) + 1
            self._doc_term_freqs.append(term_freq)
            self._doc_lengths.append(len(tokens))
            for tok in set(term_freq.keys()):
                self._df[tok] = self._df.get(tok, 0) + 1
                self._vocab.add(tok)
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self._openai is None:
            raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")
        # OpenAI API returns list of data objects with embedding vectors
        resp = self._openai.embeddings.create(model=self._embedding_model, input=texts)
        vectors = [item.embedding for item in resp.data]
        return np.array(vectors, dtype=np.float32)
    
    def build_index(self):
        if not self.chunks:
            logger.warning("No chunks to index")
            return
        texts = [c.text for c in self.chunks]
        logger.info("Building semantic embeddings (OpenAI) and keyword index...")
        # Keyword index
        self._compute_keyword_index(texts)
        # Semantic embeddings
        try:
            # Batch in reasonable sizes to avoid token limits
            batch = 256
            embs: List[np.ndarray] = []
            for i in range(0, len(texts), batch):
                batch_vecs = self._embed_texts(texts[i:i+batch])
                embs.append(batch_vecs)
            self._embeddings = np.vstack(embs)
        except Exception as e:
            logger.error(f"Error building embeddings: {e}")
            self._embeddings = None
        logger.info(f"Index built. Chunks: {len(texts)} | Embeddings: {None if self._embeddings is None else self._embeddings.shape}")
    
    def _semantic_scores(self, query: str) -> Optional[np.ndarray]:
        if self._embeddings is None:
            return None
        try:
            qvec = self._embed_texts([query])[0]
            # cosine similarity
            A = self._embeddings
            denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(qvec) + 1e-12)
            sims = (A @ qvec) / denom
            return sims
        except Exception as e:
            logger.error(f"Semantic scoring failed: {e}")
            return None
    
    def _keyword_scores(self, query: str) -> np.ndarray:
        # Simple IDF-weighted sum of term frequencies
        tokens = re.findall(r"[a-zA-Z0-9$%]+", query.lower())
        N = max(1, len(self._doc_term_freqs))
        scores = np.zeros(N, dtype=np.float32)
        for t in tokens:
            df = self._df.get(t, 0)
            if df == 0:
                continue
            idf = log((N + 1) / (df + 1)) + 1.0
            for i, tfmap in enumerate(self._doc_term_freqs):
                tf = tfmap.get(t, 0)
                if tf:
                    scores[i] += tf * idf
        # Normalize by document length to reduce bias
        lengths = np.array([max(1, L) for L in self._doc_lengths], dtype=np.float32)
        scores = scores / lengths
        return scores
    
    def search(self, query: str, top_k: int = 5, company_filter: str = None) -> List[DocumentChunk]:
        if not self.chunks:
            return []
        sem = self._semantic_scores(query)
        key = self._keyword_scores(query)
        if sem is None:
            combined = key
        else:
            # Scale to comparable ranges
            sem_norm = (sem - sem.min()) / (sem.max() - sem.min() + 1e-12)
            key_norm = (key - key.min()) / (key.max() - key.min() + 1e-12)
            combined = 0.7 * sem_norm + 0.3 * key_norm
        indices = np.argsort(combined)[::-1]
        results: List[DocumentChunk] = []
        for idx in indices:
            if len(results) >= top_k:
                break
            chunk = self.chunks[idx]
            if company_filter and chunk.company != company_filter:
                continue
            results.append(chunk)
        return results

class FinancialAgent:
    """Agent for query decomposition and multi-step reasoning"""
    
    def __init__(self, vector_store: VectorStore, llm_client):
        self.vector_store = vector_store
        self.llm = llm_client
        self.max_iterations = 3
        self.confidence_threshold = 0.75
    
    def is_complex_query(self, query: str) -> bool:
        """Determine if query needs decomposition"""
        complex_indicators = [
            'compare', 'comparison', 'all three', 'across', 'between',
            'growth', 'change', 'from', 'to', 'highest', 'lowest',
            'better', 'worse', 'increase', 'decrease'
        ]
        
        # Ensure query is a string and convert to lowercase safely
        if not isinstance(query, str):
            logger.warning(f"Query is not a string: {query} (type: {type(query)})")
            return False
            
        try:
            query_lower = str(query).lower()
            return any(indicator in query_lower for indicator in complex_indicators)
        except Exception as e:
            logger.error(f"Error in is_complex_query: {e}")
            return False
    
    def decompose_query(self, query: str) -> List[str]:
        """Break down complex query into sub-queries"""
        try:
            prompt = f"""
            Break down this financial query into specific sub-queries that can be answered individually.
            Focus on extracting specific metrics for each company and year mentioned.
            
            Query: {query}
            
            Return ONLY a JSON list of sub-queries, no other text or formatting.
            Example: ["NVIDIA total revenue 2024", "Microsoft total revenue 2024", "Google total revenue 2024"]
            """
            
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0
                )
            )
            
            result = response.text.strip()
            
            # Clean up the response - remove markdown formatting and extra characters
            result = result.replace('```json', '').replace('```', '').strip()
            result = result.replace('```', '').strip()
            
            # Try to parse as JSON
            try:
                sub_queries = json.loads(result)
                if isinstance(sub_queries, list):
                    # Clean up each sub-query
                    cleaned_queries = []
                    for q in sub_queries:
                        if isinstance(q, str):
                            # Remove extra quotes and commas
                            clean_q = q.strip().strip('"').strip("'").strip(',')
                            if clean_q and not clean_q.startswith('[') and not clean_q.startswith(']'):
                                cleaned_queries.append(clean_q)
                    return cleaned_queries if cleaned_queries else [query]
                else:
                    return [query]
            except json.JSONDecodeError:
                # Fallback: extract queries from text lines
                lines = result.split('\n')
                cleaned_queries = []
                for line in lines:
                    line = line.strip().strip('"').strip("'").strip(',')
                    # Remove JSON array markers and empty lines
                    if (line and not line.startswith('[') and not line.startswith(']') 
                        and not line.startswith('{') and not line.startswith('}')):
                        cleaned_queries.append(line)
                return cleaned_queries if cleaned_queries else [query]
                
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [query]
    
    def answer_simple_query(self, query: str) -> Dict[str, Any]:
        """Answer a simple query with single retrieval"""
        # Debug logging
        logger.info(f"Processing query in answer_simple_query: {query} (type: {type(query)})")
        
        # Validate query input
        if not query or not isinstance(query, str):
            logger.warning(f"Invalid query: {query} (type: {type(query)})")
            return {
                "answer": "Invalid query format.",
                "sources": []
            }
        
        # Clean up query
        query = query.strip()
        if not query:
            return {
                "answer": "Empty query.",
                "sources": []
            }
        
        # Search for relevant chunks
        try:
            chunks = self.vector_store.search(query, top_k=3)
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return {
                "answer": f"Error searching documents: {e}",
                "sources": []
            }
        
        if not chunks:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }
        
        # Combine context
        try:
            context = "\n\n".join([f"From {chunk.company} {chunk.year}: {chunk.text[:500]}" 
                                  for chunk in chunks])
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return {
                "answer": f"Error processing document chunks: {e}",
                "sources": []
            }
        
        # Generate answer
        try:
            prompt = f"""
            Answer this financial question based on the provided context from 10-K filings.
            Be specific with numbers and cite which company/year the information comes from.
            
            Question: {query}
            
            Context:
            {context}
            
            Answer:
            """
            
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0
                )
            )
            
            answer = response.text.strip()
            
            # Format sources
            sources = []
            for chunk in chunks:
                sources.append({
                    "company": chunk.company,
                    "year": chunk.year,
                    "excerpt": chunk.text[:200] + "...",
                    "page": chunk.page
                })
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {e}",
                "sources": []
            }
    
    def _plan(self, query: str) -> List[str]:
        """Plan node: LLM-based decomposition of the task into sub-queries."""
        return self.decompose_query(query)

    def _act(self, sub_queries: List[str]) -> List[Dict[str, Any]]:
        """Act node: retrieve/context/answers per sub-query."""
        results: List[Dict[str, Any]] = []
        for sq in sub_queries:
            if not sq or not isinstance(sq, str):
                continue
            r = self.answer_simple_query(sq)
            results.append({
                "query": sq,
                "answer": r.get("answer", ""),
                "sources": r.get("sources", [])
            })
        return results

    def _reflect(self, original_question: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reflect node: assess confidence, propose follow-ups. Returns dict with keys
        {"confidence": float 0..1, "follow_ups": [str], "final_ready": bool}.
        """
        try:
            prompt = f"""
            You are verifying an answer synthesis process for a financial Q&A agent.
            Given the original question and sub-query answers (with sources), evaluate:
            - Is the evidence sufficient to confidently answer? Return a confidence in [0,1].
            - If missing numbers or comparisons, propose targeted FOLLOW-UP sub-queries.
            Return ONLY valid JSON of the form:
            {{"confidence": <float 0..1>, "follow_ups": [<strings>], "final_ready": <true|false>}}.

            Original Question: {original_question}
            Sub-query Answers:\n{json.dumps(results, indent=2)}
            """
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=250,
                    temperature=0
                )
            )
            raw = response.text.strip().replace('```json', '').replace('```', '').strip()
            data = json.loads(raw)
            conf = float(max(0.0, min(1.0, data.get("confidence", 0.0))))
            follow = data.get("follow_ups", [])
            if not isinstance(follow, list):
                follow = []
            follow = [s for s in follow if isinstance(s, str)]
            final_ready = bool(data.get("final_ready", False))
            return {"confidence": conf, "follow_ups": follow, "final_ready": final_ready}
        except Exception:
            return {"confidence": 0.0, "follow_ups": [], "final_ready": False}

    def answer_complex_query(self, query: str) -> QueryResult:
        """Answer complex query using a Plan→Act→Reflect loop."""
        all_sources: List[Dict[str, Any]] = []
        seen_queries: set = set()
        aggregate_results: List[Dict[str, Any]] = []

        # Iterative loop
        planned = self._plan(query)
        for iteration in range(self.max_iterations):
            # Deduplicate planned sub-queries
            to_run = [sq for sq in planned if isinstance(sq, str) and sq not in seen_queries]
            seen_queries.update(to_run)
            if not to_run and iteration > 0:
                break
            # Act
            new_results = self._act(to_run)
            for r in new_results:
                aggregate_results.append(r)
                all_sources.extend(r.get("sources", []))
            # Reflect
            reflect_out = self._reflect(query, aggregate_results)
            if reflect_out.get("final_ready") or reflect_out.get("confidence", 0.0) >= self.confidence_threshold:
                break
            planned = reflect_out.get("follow_ups", [])

        # Synthesis
        try:
            synthesis_context = "\n\n".join([f"Q: {r['query']}\nA: {r['answer']}" for r in aggregate_results])
            synthesis_prompt = f"""
            Provide a final answer to the original question using the sub-query results. Where appropriate, compute
            growth rates, margins, and make cross-company comparisons. Be explicit and concise, and include figures
            with company and year labels. Ensure every numeric claim is supported by the sub-results.

            Original Question: {query}
            Sub-query Results:
            {synthesis_context}

            Final Answer:
            """
            response = self.llm.generate_content(
                synthesis_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=700,
                    temperature=0
                )
            )
            final_answer = response.text.strip()
        except Exception:
            final_answer = self._fallback_synthesize_answer(query, aggregate_results)

        return QueryResult(
            query=query,
            answer=final_answer,
            reasoning=f"Plan→Act→Reflect over {len(seen_queries)} sub-queries in ≤{self.max_iterations} iterations",
            sub_queries=[r['query'] for r in aggregate_results],
            sources=all_sources[:10]
        )
    
    def _fallback_synthesize_answer(self, query: str, sub_results: List[Dict]) -> str:
        """Synthesize answer without LLM"""
        query_lower = query.lower()
        
        if not sub_results:
            return "No information found to answer the query."
        
        # Collect all answers
        answers = [r['answer'] for r in sub_results if r['answer'] != "No relevant information found."]
        
        if not answers:
            return "No specific information found in the 10-K filings for this query."
        
        # Different synthesis strategies based on query type
        if 'compare' in query_lower or 'across' in query_lower:
            return f"Comparison analysis: {' | '.join(answers)}"
        elif 'highest' in query_lower or 'which' in query_lower:
            return f"Analysis of companies: {' '.join(answers)}"
        elif 'grow' in query_lower or 'growth' in query_lower:
            return f"Growth analysis: {answers[0] if answers else 'Growth data not clearly available'}"
        else:
            return f"Financial analysis: {answers[0] if answers else 'Information found but needs manual review'}"
        
        return QueryResult(
            query=query,
            answer=final_answer,
            reasoning=f"Decomposed into {len(sub_queries)} sub-queries and synthesized results",
            sub_queries=sub_queries,
            sources=all_sources[:10]  # Limit sources
        )
    
    def answer_query(self, query: str) -> QueryResult:
        """Main entry point for answering queries"""
        logger.info(f"Processing query: {query} (type: {type(query)})")
        
        # Ensure query is a string
        if not isinstance(query, str):
            logger.error(f"Query is not a string: {query} (type: {type(query)})")
            return QueryResult(
                query=str(query),
                answer="Error: Invalid query format",
                reasoning="Query validation failed",
                sub_queries=[],
                sources=[]
            )
        
        try:
            # Always use LLM-driven decomposition + synthesis for consistent reasoning
            logger.info("Routing all queries through agentic decomposition + synthesis")
            return self.answer_complex_query(query)
        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return QueryResult(
                query=query,
                answer=f"Error processing query: {e}",
                reasoning="Error occurred during processing",
                sub_queries=[],
                sources=[]
            )

class FinancialRAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, gemini_api_key: str, openai_api_key: Optional[str] = None):
        self.downloader = SECFilingDownloader()
        self.processor = TextProcessor()
        # Init OpenAI client for embeddings
        self._openai_client = None
        if openai_api_key and OpenAI is not None:
            try:
                os.environ['OPENAI_API_KEY'] = openai_api_key
                self._openai_client = OpenAI()
            except Exception as e:
                logger.warning(f"Failed to init OpenAI client: {e}")
        self.vector_store = VectorStore(openai_client=self._openai_client)
        # Initialize Gemini client for reasoning
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-1.5-flash')
        self.agent = FinancialAgent(self.vector_store, self.llm)
    
    def setup(self):
        """Download filings and build index"""
        logger.info("Setting up Financial RAG System...")
        
        # Download filings
        logger.info("Downloading SEC filings...")
        filings = self.downloader.download_all_filings()
        
        # Process and index filings
        logger.info("Processing and indexing documents...")
        for company, years in filings.items():
            for year, filepath in years.items():
                chunks = self.processor.process_filing(filepath, company, year)
                self.vector_store.add_chunks(chunks)
        
        # Build search index
        self.vector_store.build_index()
        
        logger.info("Setup complete!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return structured result"""
        result = self.agent.answer_query(question)
        return asdict(result)

def main():
    """Main application entry point"""
    # Load Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: Please set GEMINI_API_KEY environment variable")
        return
    openai_key = os.getenv('OPENAI_API_KEY')
    
    # Initialize system
    rag_system = FinancialRAGSystem(api_key, openai_api_key=openai_key)
    
    # Setup (download and index documents)
    rag_system.setup()
    
    # Test queries
    test_queries = [
        "What was NVIDIA's total revenue in fiscal year 2024?",
        # "What percentage of Google's 2023 revenue came from advertising?",
        # "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
        "Which of the three companies had the highest gross margin in 2023?",
        "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
    ]
    
    print("\n" + "="*80)
    print("FINANCIAL RAG SYSTEM - DEMO")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 60)
        
        try:
            result = rag_system.query(query)
            
            print(f"Answer: {result['answer']}")
            print(f"Reasoning: {result['reasoning']}")
            
            if len(result['sub_queries']) > 1:
                print(f"Sub-queries: {', '.join(result['sub_queries'])}")
            
            print(f"Sources: {len(result['sources'])} document chunks")
            
            # Pretty print JSON result
            print("\nFull JSON Response:")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    main()