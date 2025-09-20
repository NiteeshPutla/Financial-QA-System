# Financial RAG System - Design Document

## Chunking Strategy

**Approach**: Semantic chunking with 500-token chunks and 50-token overlap
- **Rationale**: 500 tokens provide sufficient context for financial metrics while remaining focused
- **Overlap**: 50-token overlap ensures continuity across chunk boundaries
- **Implementation**: Word-based splitting with page estimation for source tracking

## Embedding Model Choice

**Model**: OpenAI `text-embedding-3-large` for semantic retrieval + keyword fallback
- **Why**:
  - Strong semantic recall for finance phrases beyond exact terms
  - Robust across tables and narrative sections
  - Combined with lightweight keyword scoring for precision on exact metrics
- **Fusion**: 70% semantic cosine similarity + 30% IDF-weighted keyword score
- **Note**: Requires `OPENAI_API_KEY`; if missing, only keyword scoring runs

## Agent/Query Decomposition Approach

**Strategy**: Prompt-based decomposition using Gemini API
- **Detection**: Keyword-based complex query detection (compare, growth, across, etc.)
- **Decomposition**: LLM breaks complex queries into specific sub-queries
- **Synthesis**: LLM combines sub-query results into coherent final answers
- **Fallback**: Rule-based synthesis if LLM fails

**Example Flow**:
```
Query: "Compare R&D spending across all three companies in 2023"
↓
Sub-queries: ["Microsoft R&D spending 2023", "Google R&D spending 2023", "NVIDIA R&D spending 2023"]
↓
Parallel retrieval for each sub-query
↓
Synthesis: "Microsoft spent $27.2B (13.1% of revenue), Google spent $39.5B (11.8%), NVIDIA spent $7.3B (13.2%)"
```

## Interesting Challenges/Decisions

### 1. Array Comparison Error
**Challenge**: NumPy array boolean evaluation causing "ambiguous truth value" errors
**Solution**: Changed `if not self.vectors` to `if self.vectors is None` to avoid array boolean conversion

### 2. Gemini Response Format
**Challenge**: Gemini API returns malformed JSON with markdown formatting
**Solution**: Robust parsing with multiple fallback strategies:
- Remove markdown formatting (`\`\`\`json`)
- Clean extra quotes and commas
- Fallback to line-by-line text extraction

### 3. Query Validation
**Challenge**: Ensuring all inputs are valid strings before processing
**Solution**: Comprehensive type checking and validation at each processing stage

### 4. Error Handling
**Challenge**: System should gracefully handle failures without crashing
**Solution**: Try-catch blocks around all major operations with meaningful error messages

## Architecture Decisions

### Modular Design
- Separated concerns into distinct classes (Downloader, Processor, VectorStore, Agent)
- Clean interfaces between components
- Easy to test and modify individual components

### In-Memory Vector Store
- **Pros**: Simple, fast, minimal dependencies; embeds once at startup
- **Cons**: Limited scalability, no persistence
- **Decision**: Appropriate for this scope; could migrate to FAISS/Chroma later

### HTML & Table Parsing
- **Choice**: BeautifulSoup for HTML parsing and table extraction
- **Tables → JSON**: Each `<table>` is parsed to JSON rows and written to `data/{COMPANY}_{YEAR}_tables.json`
- **Embedding Tables**: Each row is converted to readable text like `TABLE {title} | col: value | ...` and embedded with metadata `section="table:{title}#row{n}`

## Performance Considerations

- **Chunking**: 500-token chunks balance context vs. precision
- **Vector Search**: Cosine similarity with TF-IDF is fast and effective
- **Caching**: Downloaded filings are cached to avoid re-downloading
- **Parallel Processing**: Sub-queries are processed independently for efficiency

## Future Improvements

1. **Persistent Vector Store**: Add FAISS/Chroma for disk-backed indices
2. **Section-aware Chunking**: Preserve MD&A, Item 8 boundaries as metadata
3. **Metric Extractors**: Regex/NER to normalize numeric values and units
4. **Tool-using Agent**: Add calculator tool; ensure verified numeric ops
5. **Caching**: Cache per-subquery results to speed repeated runs
