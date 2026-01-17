import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
from datetime import datetime
class QuoteRAGSystem:
    def __init__(self, use_llm=False, api_key=None, use_groq=True):
        print("Initializing Quote RAG System...")
        print("Loading sentence transformer...")
        self.model = SentenceTransformer('./quote_model')
        print("Loading FAISS index...")
        self.index = faiss.read_index('faiss_index.bin')
        print("Loading quote metadata...")
        with open('quote_data.pkl', 'rb') as f:
            data = pickle.load(f)
        self.quotes = data['quotes']
        self.authors = data['authors']
        self.tags = data['tags']
        self.embeddings = data['embeddings']
        self.use_llm = use_llm
        self.use_groq = use_groq
        if use_llm:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                if use_groq:
                    api_key = api_key or os.getenv('GROQ_API_KEY')
                    if api_key:
                        self.client = OpenAI(
                            api_key=api_key,
                            base_url="https://api.groq.com/openai/v1"
                        )
                        print("✓ Groq API client initialized (Fast LLM inference)")
                    else:
                        print("⚠ No Groq API key found, will use rule-based generation")
                        self.use_llm = False
                else:
                    api_key = api_key or os.getenv('OPENAI_API_KEY')
                    if api_key:
                        self.client = OpenAI(api_key=api_key)
                        print("✓ OpenAI client initialized")
                    else:
                        print("⚠ No API key found, will use rule-based generation")
                        self.use_llm = False
            except Exception as e:
                print(f"⚠ Could not initialize LLM: {e}")
                self.use_llm = False
        print("✓ RAG System ready!\n")
    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            tags = self.tags[idx]
            if isinstance(tags, str):
                try:
                    tags = eval(tags)
                except:
                    tags = []
            result = {
                'rank': rank,
                'quote': self.quotes[idx],
                'author': self.authors[idx],
                'tags': tags if isinstance(tags, list) else [],
                'score': float(score)
            }
            results.append(result)
        return results
    def generate_summary_rule_based(self, query, results):
        top_result = results[0]
        num_results = len(results)
        authors = list(set([r['author'] for r in results]))
        summary = f"Found {num_results} relevant quotes"
        if len(authors) <= 3:
            summary += f" from {', '.join(authors)}"
        summary += f". The most relevant quote (score: {top_result['score']:.3f}) is "
        summary += f'"{top_result["quote"][:100]}..." by {top_result["author"]}.'
        all_tags = []
        for r in results:
            all_tags.extend(r['tags'][:2])
        if all_tags:
            common_tags = list(set(all_tags))[:3]
            summary += f" Common themes: {', '.join(common_tags)}."
        return summary
    def generate_summary_llm(self, query, results):
        if not self.use_llm:
            return self.generate_summary_rule_based(query, results)
        context = "Retrieved Quotes:\n"
        for r in results[:3]:
            context += f"\n{r['rank']}. \"{r['quote']}\" - {r['author']}\n"
            if r['tags']:
                context += f"   Tags: {', '.join(r['tags'][:3])}\n"
        try:
            model = "llama-3.1-70b-versatile" if self.use_groq else "gpt-3.5-turbo"
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes quotes and provides insightful analysis. Be concise and engaging."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}, falling back to rule-based")
            return self.generate_summary_rule_based(query, results)
    def query(self, query_text, top_k=5, use_llm=False):
        results = self.retrieve(query_text, top_k)
        if use_llm and self.use_llm:
            summary = self.generate_summary_llm(query_text, results)
        else:
            summary = self.generate_summary_rule_based(query_text, results)
        response = {
            'query': query_text,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': summary
        }
        return response
def demo_rag_system():
    print("="*60)
    print("RAG PIPELINE DEMO")
    print("="*60)
    rag = QuoteRAGSystem(use_openai=False)
    test_queries = [
        "quotes about hope by Oscar Wilde",
        "inspirational quotes about success and hard work",
        "funny quotes about mistakes and learning",
        "love and romance quotes",
        "wisdom about life and happiness"
    ]
    print("\nRunning test queries...\n")
    for query in test_queries:
        print("="*60)
        print(f"Query: {query}")
        print("="*60)
        response = rag.query(query, top_k=3, use_llm=False)
        print(f"\nSummary: {response['summary']}\n")
        for result in response['results']:
            print(f"[{result['rank']}] Score: {result['score']:.4f}")
            print(f"    Quote: {result['quote'][:80]}...")
            print(f"    Author: {result['author']}")
            if result['tags']:
                print(f"    Tags: {', '.join(result['tags'][:3])}")
            print()
    print("\nSaving sample response to 'sample_rag_response.json'...")
    sample_response = rag.query("quotes about hope by Oscar Wilde", top_k=5)
    with open('sample_rag_response.json', 'w') as f:
        json.dump(sample_response, f, indent=2)
    print("✓ Sample response saved!")
    print("\n" + "="*60)
    print("RAG PIPELINE DEMO COMPLETE")
    print("="*60)
    print("\nThe system can:")
    print("  ✓ Retrieve semantically similar quotes")
    print("  ✓ Rank by relevance score")
    print("  ✓ Generate contextual summaries")
    print("  ✓ Support OpenAI integration (optional)")
    print("\nTo use with OpenAI:")
    print("  1. Create .env file with OPENAI_API_KEY=your_key")
    print("  2. Run: rag = QuoteRAGSystem(use_openai=True)")
if __name__ == "__main__":
    demo_rag_system()
