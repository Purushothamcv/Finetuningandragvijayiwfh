# RAG-Based Semantic Quote Retrieval System

**Assignment**: Vijayi WFH Technologies - Task 2  
**Topic**: AI/ML - RAG System for Quote Search


A complete Retrieval Augmented Generation (RAG) system for semantic quote search using:
- **Sentence Transformers** for embeddings
- **FAISS** for vector similarity search
- **Optional LLM** integration (OpenAI GPT)
- **RAGAS** for evaluation
- **Streamlit** for web interface


```
VijayiwfhTask2/
├── data_preprocessing.py       # Step 1: Load and clean data
├── create_vector_db.py          # Step 2-3: Create embeddings + FAISS
├── rag_pipeline.py              # Step 4: RAG implementation
├── evaluate_rag.py              # Step 5: Evaluation with RAGAS
├── streamlit_app.py             # Step 6: Web UI
├── requirements.txt             # Dependencies
└── README.md                    # This file
```



### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Step 1: Preprocess data
python data_preprocessing.py

# Step 2-3: Create vector database
python create_vector_db.py

# Step 4: Test RAG pipeline
python rag_pipeline.py

# Step 5: Evaluate system
python evaluate_rag.py

# Step 6: Launch web app
streamlit run streamlit_app.py
```

## 📊 System Architecture

```
User Query
    ↓
[Sentence Transformer]
    ↓
Query Embedding
    ↓
[FAISS Vector Search] → Retrieve Top-K Similar Quotes
    ↓
Retrieved Contexts
    ↓
[LLM / Rule-Based Generator] → Generate Summary
    ↓
Structured Response (JSON)
```

## 🔧 Implementation Details

### Step 1: Data Preprocessing
- **Dataset**: Abirate/english_quotes (HuggingFace)
- **Records**: 2,508 quotes
- **Processing**:
  - Text normalization (lowercase, trim spaces)
  - Combine quote + author + tags
- **Output**: `quotes_preprocessed.csv`

### Step 2-3: Embeddings & Vector DB
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Embedding Strategy**: Sentence-level semantic encoding
- **Vector Store**: FAISS IndexFlatIP (cosine similarity)
- **Output**: `faiss_index.bin`, `quote_model/`, `quote_data.pkl`

### Step 4: RAG Pipeline
- **Retrieval**: Top-K semantic search
- **Generation**: 
  - Rule-based (fast, no API needed)
  - LLM-based (OpenAI GPT-3.5/4, optional)
- **Output Format**: JSON with query, results, and summary

```json
{
  "query": "quotes about hope",
  "results": [
    {
      "rank": 1,
      "quote": "Hope is the thing with feathers...",
      "author": "Emily Dickinson",
      "tags": ["hope", "inspiration"],
      "score": 0.8543
    }
  ],
  "summary": "Found 5 relevant quotes..."
}
```

### Step 5: Evaluation
- **Framework**: RAGAS
- **Metrics**:
  - Context Precision
  - Context Recall
  - Faithfulness
  - Answer Relevancy
- **Output**: `evaluation_results.json`, `evaluation_report.md`

### Step 6: Streamlit UI
- **Features**:
  - Search interface
  - Adjustable top-K results
  - Display with scores
  - JSON/TXT export
- **Launch**: `streamlit run streamlit_app.py`



### Python API

```python
from rag_pipeline import QuoteRAGSystem

# Initialize system
rag = QuoteRAGSystem(use_openai=False)

# Search
response = rag.query("quotes about hope by Oscar Wilde", top_k=5)

# Access results
for result in response['results']:
    print(f"{result['quote']} - {result['author']}")
```

### CLI

```bash
# Quick search
python -c "from rag_pipeline import QuoteRAGSystem; \
           rag = QuoteRAGSystem(); \
           print(rag.query('inspirational quotes')['summary'])"
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Total Quotes | 2,508 |
| Embedding Dimension | 384 |
| Search Time | <100ms |
| Model Size | ~90MB |
| Index Size | ~3MB |

## 🔑 Optional: OpenAI Integration

For enhanced summaries using GPT:

1. Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

2. Use in code:
```python
rag = QuoteRAGSystem(use_openai=True)
response = rag.query("your query", use_llm=True)
```

## 🧪 Testing

```bash
# Test data preprocessing
python data_preprocessing.py

# Test vector DB creation
python create_vector_db.py

# Test RAG pipeline
python rag_pipeline.py

# Run evaluation
python evaluate_rag.py
```

## 📝 Key Features

✅ **Semantic Search**: Understanding query intent, not just keywords  
✅ **Fast Retrieval**: FAISS enables millisecond search  
✅ **Flexible Generation**: Works with/without LLM  
✅ **Evaluation Framework**: RAGAS metrics for quality assessment  
✅ **Web Interface**: User-friendly Streamlit app  
✅ **Export Options**: JSON and text formats  

## 🎓 Interview Talking Points

### 1. **Why RAG over fine-tuning?**
- RAG provides up-to-date information without retraining
- Combines retrieval accuracy with generation fluency
- More interpretable (can trace sources)

### 2. **Model Selection**
- MiniLM: Lightweight (384-dim) yet effective
- Good balance between speed and accuracy
- Proven performance on semantic similarity tasks

### 3. **FAISS vs Alternatives**
- FAISS: Production-ready, highly optimized
- Supports billions of vectors
- Multiple index types for different use cases

### 4. **Evaluation Strategy**
- RAGAS provides comprehensive RAG-specific metrics
- Measures both retrieval and generation quality
- Enables continuous improvement tracking

### 5. **Production Considerations**
- Caching for frequent queries
- Batch processing for efficiency
- Monitoring and logging
- A/B testing framework

## 🔄 Future Enhancements

1. **Fine-tuning**: Domain-specific model training
2. **Query Expansion**: Synonyms and paraphrases
3. **Re-ranking**: Two-stage retrieval pipeline
4. **Multi-modal**: Support images/audio
5. **User Feedback**: Click data for improvement

## 📚 Dependencies

Core libraries:
- `sentence-transformers==2.5.1`
- `faiss-cpu`
- `streamlit`
- `datasets`
- `pandas`
- `numpy`
- `ragas`
- `openai` (optional)
- `python-dotenv` (optional)

## 🐛 Troubleshooting

### Issue: Model download slow
**Solution**: Pre-download model:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
```

### Issue: FAISS import error
**Solution**: Install CPU version:
```bash
pip install faiss-cpu
```


