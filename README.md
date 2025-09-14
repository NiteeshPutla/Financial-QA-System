# Financial Q&A System with Agent Capabilities

A focused RAG system with agent capabilities that can answer both simple and comparative financial questions about Google, Microsoft, and NVIDIA using their recent 10-K filings. The system demonstrates query decomposition and multi-step reasoning for complex questions.

## Features

- **Automated SEC Filing Download**: Downloads 10-K filings for GOOGL, MSFT, and NVDA (2022-2024)
- **Query Decomposition**: Breaks complex questions into sub-queries
- **Multi-step Reasoning**: Executes multiple searches to answer comparative questions
- **Vector-based RAG**: TF-IDF embeddings with cosine similarity search
- **Agent Orchestration**: Synthesizes results from multiple retrievals into coherent answers

## Project Structure

```
Financial QA System/
├── finance.py                 # Main system implementation
├── requirements.txt           # Python dependencies
├── data/                      # Downloaded SEC filings
│   ├── GOOGL_2022_10K.html
│   ├── GOOGL_2023_10K.html
│   ├── GOOGL_2024_10K.html
│   ├── MSFT_2022_10K.html
│   ├── MSFT_2023_10K.html
│   ├── MSFT_2024_10K.html
│   ├── NVDA_2022_10K.html
│   ├── NVDA_2023_10K.html
│   └── NVDA_2024_10K.html
└── README.md                  # This file
```

## System Architecture

The system is built with a modular architecture that separates concerns:

### Core Components

1. **SECFilingDownloader**: Downloads 10-K filings from SEC EDGAR database
2. **TextProcessor**: Extracts and chunks text from HTML filings
3. **VectorStore**: TF-IDF based document storage and similarity search
4. **FinancialAgent**: Query decomposition and multi-step reasoning
5. **FinancialRAGSystem**: Main orchestrator that coordinates all components

### Design Choices

- **Chunking Strategy**: 500-token chunks with 50-token overlap for optimal context
- **Embedding Model**: TF-IDF for simplicity and effectiveness on financial text
- **Vector Storage**: In-memory implementation with cosine similarity
- **LLM Integration**: Google Gemini API for query decomposition and synthesis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NiteeshPutla/Financial-QA-System.git
cd Financial-QA-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
```bash
# Create a .env file
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```


## Usage

### Quick Start
```bash
python finance.py
```

## Supported Query Types

The system supports all 5 required query types from the assignment:

1. **Basic Metrics**: "What was Microsoft's total revenue in 2023?"
2. **YoY Comparison**: "How did NVIDIA's data center revenue grow from 2022 to 2023?"
3. **Cross-Company**: "Which company had the highest operating margin in 2023?"
4. **Segment Analysis**: "What percentage of Google's revenue came from cloud in 2023?"
5. **AI Strategy**: "Compare AI investments mentioned by all three companies in their 2024 10-Ks"

## Sample Queries

```python
test_queries = [
    # Simple queries
    "What was NVIDIA's total revenue in fiscal year 2024?",
    "What percentage of Google's 2023 revenue came from advertising?",
    
    # Comparative queries (require agent decomposition)
    "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
    "Which of the three companies had the highest gross margin in 2023?",
    
    # Complex multi-step queries
    "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
    "How did each company's operating margin change from 2022 to 2024?",
    "What are the main AI risks mentioned by each company and how do they differ?"
]
```

## Output Format

The system returns structured JSON responses with sources:

```json
{
  "query": "Which company had the highest operating margin in 2023?",
  "answer": "Microsoft had the highest operating margin at 42.1% in 2023...",
  "reasoning": "Retrieved operating margins for all three companies from their 2023 10-K filings...",
  "sub_queries": [
    "Microsoft operating margin 2023",
    "Google operating margin 2023", 
    "NVIDIA operating margin 2023"
  ],
  "sources": [
    {
      "company": "MSFT",
      "year": "2023",
      "excerpt": "Operating margin was 42.1%...",
      "page": 10
    }
  ]
}
```

## Data Scope

- **Companies**: Google (GOOGL), Microsoft (MSFT), NVIDIA (NVDA)
- **Documents**: Annual 10-K filings only
- **Years**: 2022, 2023, 2024
- **Total Files**: 9 documents (3 companies × 3 years)
- **Source**: SEC EDGAR database

## Dependencies

- `google-generativeai>=0.3.0` - Gemini API integration
- `numpy>=1.24.0` - Numerical operations
- `scikit-learn>=1.3.0` - TF-IDF vectorization and cosine similarity
- `requests>=2.31.0` - HTTP requests for SEC API
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=4.9.0` - XML/HTML processing
- `python-dotenv>=1.0.0` - Environment variable management

## Agent Implementation

### Query Decomposition Strategy
The system uses a prompt-based approach to decompose complex queries:

```python
# Example: "Compare cloud revenue growth for all companies 2022-2023"
sub_queries = [
    "Microsoft cloud revenue 2022",
    "Microsoft cloud revenue 2023", 
    "Google cloud revenue 2022",
    "Google cloud revenue 2023",
    "NVIDIA data center revenue 2022",
    "NVIDIA data center revenue 2023"
]
```

### Multi-step Reasoning Flow
1. **Query Analysis**: Determines if query needs decomposition
2. **Sub-query Generation**: Breaks complex queries into specific metrics
3. **Parallel Retrieval**: Searches for each sub-query independently
4. **Result Synthesis**: Combines findings into coherent answers

### Key Challenges Addressed
- **Array Comparison Errors**: Fixed numpy array boolean evaluation issues
- **Malformed Responses**: Robust parsing of Gemini API responses
- **Error Handling**: Comprehensive error handling throughout the pipeline
- **Query Validation**: Input validation and type checking


