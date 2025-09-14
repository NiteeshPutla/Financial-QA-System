# Financial RAG System - Design Document

## Chunking Strategy

**Approach**: Semantic chunking with 500-token chunks and 50-token overlap
- **Rationale**: 500 tokens provide sufficient context for financial metrics while remaining focused
- **Overlap**: 50-token overlap ensures continuity across chunk boundaries
- **Implementation**: Word-based splitting with page estimation for source tracking

## Embedding Model Choice

**Model**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Why**: 
  - Simple and effective for financial text with specific terminology
  - No external API dependencies or model downloads
  - Fast computation and good performance on domain-specific vocabulary
  - Works well with the structured nature of 10-K filings
- **Alternative Considered**: OpenAI embeddings, but TF-IDF was chosen for simplicity and self-contained operation

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
- **Pros**: Simple, fast, no external dependencies
- **Cons**: Limited scalability, data lost on restart
- **Decision**: Appropriate for this scope and demonstrates core RAG concepts

### HTML Parsing
- **Choice**: BeautifulSoup for HTML parsing
- **Rationale**: Robust parsing of SEC HTML filings, handles malformed HTML well
- **Alternative**: Could use more sophisticated parsers, but BeautifulSoup is sufficient

## Performance Considerations

- **Chunking**: 500-token chunks balance context vs. precision
- **Vector Search**: Cosine similarity with TF-IDF is fast and effective
- **Caching**: Downloaded filings are cached to avoid re-downloading
- **Parallel Processing**: Sub-queries are processed independently for efficiency

## Future Improvements

1. **Better Embeddings**: Could upgrade to sentence transformers or OpenAI embeddings
2. **Persistent Vector Store**: Could add ChromaDB or FAISS for production use
3. **Table Parsing**: Could add financial table extraction for more precise metrics
4. **Query Optimization**: Could add query expansion and synonym handling
5. **Caching**: Could add result caching for repeated queries
