import os
import json
import re
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv

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
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', html_content)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {filepath}: {e}")
            return ""
    
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
        """Process a single filing into chunks"""
        text = self.extract_text_from_html(filepath)
        return self.chunk_text(text, company, year)

class VectorStore:
    """Simple in-memory vector store using TF-IDF"""
    
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.vectors: Optional[np.ndarray] = None
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the store"""
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")
    
    def build_index(self):
        """Build TF-IDF index for all chunks"""
        if not self.chunks:
            logger.warning("No chunks to index")
            return
        
        logger.info("Building TF-IDF index...")
        
        # Extract text from all chunks
        texts = [chunk.text for chunk in self.chunks]
        
        # Build TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.vectors = self.vectorizer.fit_transform(texts)
        logger.info(f"Built index with {self.vectors.shape[0]} documents")
    
    def search(self, query: str, top_k: int = 5, company_filter: str = None) -> List[DocumentChunk]:
        """Search for relevant chunks"""
        if self.vectors is None or self.vectorizer is None:
            logger.error("Index not built. Call build_index() first.")
            return []
        
        # Vectorize query
        try:
            query_vector = self.vectorizer.transform([query])
        except Exception as e:
            logger.error(f"Error vectorizing query: {e}")
            return []
        
        # Calculate similarities
        try:
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return []
        
        # Get top-k indices, ensuring we have valid similarities
        if similarities.size == 0:
            logger.warning("No similarities calculated")
            return []
            
        try:
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering
        except Exception as e:
            logger.error(f"Error sorting similarities: {e}")
            return []
        
        # Filter results
        results = []
        try:
            for idx in top_indices:
                if idx >= len(self.chunks):
                    continue
                    
                chunk = self.chunks[idx]
                
                # Apply company filter if specified
                if company_filter and chunk.company != company_filter:
                    continue
                
                if len(results) >= top_k:
                    break
                    
                results.append(chunk)
        except Exception as e:
            logger.error(f"Error filtering results: {e}")
            return []
        
        return results

class FinancialAgent:
    """Agent for query decomposition and multi-step reasoning"""
    
    def __init__(self, vector_store: VectorStore, llm_client):
        self.vector_store = vector_store
        self.llm = llm_client
    
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
    
    def answer_complex_query(self, query: str) -> QueryResult:
        """Answer complex query using decomposition"""
        # Decompose query
        sub_queries = self.decompose_query(query)
        
        logger.info(f"Decomposed query into: {sub_queries}")
        
        # Answer each sub-query
        sub_results = []
        all_sources = []
        
        for sub_query in sub_queries:
            # Validate sub-query before processing
            if not sub_query or not isinstance(sub_query, str):
                logger.warning(f"Skipping invalid sub-query: {sub_query}")
                continue
                
            try:
                result = self.answer_simple_query(sub_query)
                sub_results.append({
                    "query": sub_query,
                    "answer": result["answer"],
                    "sources": result["sources"]
                })
                all_sources.extend(result["sources"])
            except Exception as e:
                logger.error(f"Error processing sub-query '{sub_query}': {e}")
                sub_results.append({
                    "query": sub_query,
                    "answer": f"Error processing query: {e}",
                    "sources": []
                })
        
        # Synthesize final answer
        try:
            synthesis_context = "\n\n".join([
                f"Q: {r['query']}\nA: {r['answer']}" for r in sub_results
            ])
            
            synthesis_prompt = f"""
            Based on the following sub-query results, provide a comprehensive answer to the original question.
            Synthesize the information and provide specific comparisons, calculations, or insights as needed.
            
            Original Question: {query}
            
            Sub-query Results:
            {synthesis_context}
            
            Final Answer:
            """
            
            response = self.llm.generate_content(
                synthesis_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0
                )
            )
            
            final_answer = response.text.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing answer with LLM: {e}")
            # Fallback synthesis
            final_answer = self._fallback_synthesize_answer(query, sub_results)
        
        return QueryResult(
            query=query,
            answer=final_answer,
            reasoning=f"Decomposed into {len(sub_queries)} sub-queries and synthesized results",
            sub_queries=sub_queries,
            sources=all_sources[:10]  # Limit sources
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
            if self.is_complex_query(query):
                logger.info("Complex query detected, using decomposition")
                return self.answer_complex_query(query)
            else:
                logger.info("Simple query, using direct retrieval")
                result = self.answer_simple_query(query)
                return QueryResult(
                    query=query,
                    answer=result["answer"],
                    reasoning="Direct retrieval and synthesis",
                    sub_queries=[query],
                    sources=result["sources"]
                )
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
    
    def __init__(self, gemini_api_key: str):
        self.downloader = SECFilingDownloader()
        self.processor = TextProcessor()
        self.vector_store = VectorStore()
        
        # Initialize Gemini client
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
    
    # Initialize system
    rag_system = FinancialRAGSystem(api_key)
    
    # Setup (download and index documents)
    rag_system.setup()
    
    # Test queries
    test_queries = [
        "What was NVIDIA's total revenue in fiscal year 2024?",
        "What percentage of Google's 2023 revenue came from advertising?",
        "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
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