# RAG-Based Semantic Quote Retrieval System

Assignment: Vijayi WFH Technologies - Task 2
Topic: AI/ML - RAG System for Quote Search
Author: Purushotham CV

<<<<<<< HEAD
## Project Overview
=======
>>>>>>> 0cbd54e567c9a76de828c6fe90094356420c1951

A complete Retrieval Augmented Generation (RAG) system for semantic quote search that combines TF-IDF based retrieval with optional AI-powered response generation using Groq's Llama 3.1 model.

<<<<<<< HEAD
Key Features:
- TF-IDF vectorization for fast and accurate quote retrieval
- Natural language query parsing with multi-hop filtering
- AI-powered insights using Groq Llama 3.1 70B model
- Comprehensive evaluation framework with custom metrics
- Interactive Streamlit web interface

## Project Structure
=======
>>>>>>> 0cbd54e567c9a76de828c6fe90094356420c1951

```
VijayiwfhTask2/
├── data_preprocessing.py              # Data loading and cleaning
├── create_simple_search.py           # TF-IDF model training
├── rag_pipeline.py                   # RAG system with Groq integration
├── evaluate_comprehensive.py         # Evaluation metrics
├── streamlit_advanced_app.py         # Web application
├── quotes_preprocessed.csv           # Processed dataset (2,508 quotes)
├── search_system.pkl                 # Trained TF-IDF model
├── comprehensive_evaluation_results.csv  # Evaluation metrics
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

<<<<<<< HEAD
## Installation and Setup
=======

>>>>>>> 0cbd54e567c9a76de828c6fe90094356420c1951

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key (Optional for AI Features)

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Pipeline

```bash
# Step 1: Preprocess data
python data_preprocessing.py

# Step 2: Create search system
python create_simple_search.py

# Step 3: Run evaluation
python evaluate_comprehensive.py

# Step 4: Launch web application
streamlit run streamlit_advanced_app.py
```

## System Architecture

```
User Query
    ↓
Natural Language Parser (extract filters)
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity Search
    ↓
Multi-hop Filtering (author/gender/tags/century)
    ↓
Top-K Results Ranking
    ↓
AI Summary Generation (Groq Llama 3.1 - Optional)
    ↓
JSON Response + Visualizations
```

## Implementation Details

### Data Preprocessing
- Dataset: Abirate/english_quotes from HuggingFace
- Total Records: 2,508 quotes
- Processing Steps:
  - Text normalization (lowercase, whitespace handling)
  - Combined text creation (quote + author + tags)
- Output: quotes_preprocessed.csv

### Search System (TF-IDF)
- Vectorizer: TfidfVectorizer with 1,000 features
- N-gram Range: (1, 2) for better phrase matching
- Similarity Metric: Cosine similarity
- Search Speed: Under 300ms per query
- Model Size: Approximately 1.2 MB

### RAG Pipeline
- Retrieval: TF-IDF based semantic search
- Generation Options:
  - Rule-based summaries (fast, no API required)
  - AI-powered insights using Groq Llama 3.1 70B
- Multi-hop Query Support:
  - Tag filtering (ALL tags must match)
  - Century-based filtering (ancient/modern authors)
  - Gender and author filtering
- Response Format: Structured JSON with metadata

### Evaluation Metrics
Framework: Custom comprehensive evaluation + RAGAS integration

Performance Results:
- Overall Score: 73.85%
- Retrieval Quality: 57.41%
- Precision: 96.00%
- Diversity: 92.00%
- Response Quality: 67.5%

### Web Application Features
- Natural language search interface
- Advanced filtering options
- AI-powered summary toggle
- Interactive visualizations (Plotly):
  - Similarity score distribution
  - Author frequency analysis
  - Tag distribution charts
  - Dataset overview
- JSON export functionality
- Real-time search results

<<<<<<< HEAD
## Usage Examples
=======

>>>>>>> 0cbd54e567c9a76de828c6fe90094356420c1951

### Basic Search

```python
# In Python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load search system
with open('search_system.pkl', 'rb') as f:
    data = pickle.load(f)

vectorizer = data['vectorizer']
tfidf_matrix = data['tfidf_matrix']

# Search
query = "wisdom and knowledge"
query_vec = vectorizer.transform([query])
similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
top_indices = similarities.argsort()[-5:][::-1]
```

### Advanced Multi-hop Queries

Examples of supported natural language queries:
- "quotes about courage by women authors"
- "wisdom from ancient philosophers"
- "quotes tagged with 'life' and 'love' by 20th century authors"
- "inspirational quotes by Shakespeare"

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Quotes | 2,508 |
| TF-IDF Features | 1,000 |
| Search Speed | <300ms |
| Precision | 96% |
| Overall Score | 73.85% |
| Model Size | 1.2 MB |

## Key Features

- Semantic Search: TF-IDF based similarity matching
- Fast Retrieval: Optimized vector operations
- Flexible Generation: Works with or without AI API
- Comprehensive Evaluation: Multiple quality metrics
- Interactive UI: User-friendly Streamlit interface
- Export Options: JSON download functionality
- Visualizations: Interactive charts for result analysis

## Technical Decisions

### Why TF-IDF over Neural Embeddings?
- Faster training and inference
- No GPU requirements
- Interpretable feature importance
- Excellent precision (96%) on this dataset
- Smaller model size and memory footprint

### Groq vs OpenAI for LLM
- Groq provides faster inference (Llama 3.1 70B)
- Cost-effective API pricing
- High-quality natural language generation
- OpenAI-compatible API interface

### Evaluation Approach
- Combined RAGAS framework with custom metrics
- Measures both retrieval and generation quality
- Tracks precision, diversity, and relevance
- Provides actionable insights for improvement

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection for initial data download
- Groq API key (optional, for AI features)

## Dependencies

Core libraries:
- scikit-learn
- pandas
- numpy
- streamlit
- plotly
- datasets (HuggingFace)
- ragas
- python-dotenv
- openai (for Groq API compatibility)

## Troubleshooting

### Issue: Dataset download fails
Solution: Check internet connection and HuggingFace dataset availability

### Issue: Model file not found
Solution: Run create_simple_search.py to generate search_system.pkl

<<<<<<< HEAD
### Issue: AI summaries not working
Solution: Verify GROQ_API_KEY is set correctly in .env file

### Issue: Streamlit port already in use
Solution: Use `streamlit run streamlit_advanced_app.py --server.port=8502`

## Future Enhancements

1. Neural embedding models for improved semantic understanding
2. Query expansion with synonyms
3. User feedback integration for relevance tuning
4. Caching layer for frequent queries
5. Multi-language support
6. Advanced re-ranking algorithms

## Project Evaluation Results

### Retrieval Performance
- Top-1 Accuracy: 88.24%
- Top-5 Precision: 96%
- Average Similarity Score: 73.85%

### System Capabilities
- Handles complex multi-hop queries
- Supports natural language input
- Real-time filtering and ranking
- AI-enhanced response generation

## License

Educational project for interview assignment.

## Author

Purushotham CV
Assignment submission for Vijayi WFH Technologies
Date: January 2026

## Acknowledgments

- Dataset: Abirate/english_quotes (HuggingFace)
- AI Model: Groq Llama 3.1 70B
- Framework: Streamlit for web interface
- Evaluation: RAGAS framework
=======

>>>>>>> 0cbd54e567c9a76de828c6fe90094356420c1951
