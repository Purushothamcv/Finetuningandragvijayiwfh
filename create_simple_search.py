import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
print("="*60)
print("Creating Simple Search System (TF-IDF Based)")
print("="*60)
print("\nStep 1: Loading data...")
df = pd.read_csv('quotes_preprocessed.csv')
print(f"✓ Loaded {len(df)} quotes")
print("\nStep 2: Creating TF-IDF vectors...")
texts = df['combined_text'].fillna('').tolist()
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(texts)
print(f"✓ Created TF-IDF matrix: {tfidf_matrix.shape}")
print("\nStep 3: Saving artifacts...")
data = {
    'tfidf_matrix': tfidf_matrix,
    'vectorizer': vectorizer,
    'quotes': df['quote'].tolist(),
    'authors': df['author'].tolist(),
    'tags': df['tags'].tolist(),
    'combined_text': df['combined_text'].tolist()
}
with open('search_system.pkl', 'wb') as f:
    pickle.dump(data, f)
print("✓ Saved: search_system.pkl")
print("\n" + "="*60)
print("SUCCESS! Search system created!")
print("="*60)
print("\n" + "="*60)
print("Testing Search System")
print("="*60)
test_query = "wisdom and knowledge"
print(f"\nQuery: '{test_query}'")
query_vec = vectorizer.transform([test_query])
similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
top_indices = similarities.argsort()[-5:][::-1]
print(f"\nTop 5 Results:")
print("-" * 60)
for i, idx in enumerate(top_indices, 1):
    quote = df.iloc[idx]
    print(f"\n{i}. Score: {similarities[idx]:.4f}")
    print(f"   Quote: {quote['quote'][:100]}...")
    print(f"   Author: {quote['author']}")
    print(f"   Tags: {quote['tags']}")
print("\n" + "="*60)
print("You can now run: streamlit run streamlit_advanced_app.py")
print("="*60)
