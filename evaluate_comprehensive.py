import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
print("="*70)
print(" " * 15 + "COMPREHENSIVE RAG EVALUATION")
print("="*70)
print("\nLoading search system...")
with open('search_system.pkl', 'rb') as f:
    data = pickle.load(f)
vectorizer = data['vectorizer']
tfidf_matrix = data['tfidf_matrix']
quotes = data['quotes']
authors = data['authors']
tags = data['tags']
combined_texts = data['combined_text']
print(f"✓ Loaded {len(quotes)} quotes")
def search_quotes(query, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'quote': quotes[idx],
            'author': authors[idx],
            'tags': tags[idx],
            'similarity': float(similarities[idx]),
            'combined_text': combined_texts[idx]
        })
    return results
test_cases = [
    {
        'query': 'wisdom and knowledge',
        'expected_themes': ['wisdom', 'knowledge', 'learning', 'understanding'],
        'expected_authors': ['socrates', 'einstein', 'confucius'],
        'description': 'Philosophical quotes about wisdom and knowledge'
    },
    {
        'query': 'love and relationships',
        'expected_themes': ['love', 'relationship', 'heart', 'romance'],
        'expected_authors': ['shakespeare', 'wilde'],
        'description': 'Romantic quotes about love'
    },
    {
        'query': 'courage and strength',
        'expected_themes': ['courage', 'strength', 'brave', 'fear'],
        'expected_authors': ['churchill', 'roosevelt'],
        'description': 'Inspirational quotes about courage'
    },
    {
        'query': 'success and failure',
        'expected_themes': ['success', 'failure', 'achieve', 'goal'],
        'expected_authors': ['edison', 'jobs'],
        'description': 'Motivational quotes about success'
    },
    {
        'query': 'truth and honesty',
        'expected_themes': ['truth', 'honest', 'lie', 'integrity'],
        'expected_authors': ['gandhi', 'twain'],
        'description': 'Quotes about truth and honesty'
    },
]
print(f"\n✓ Created {len(test_cases)} test cases")
evaluation_results = []
print("\n" + "="*70)
print(" " * 20 + "RUNNING EVALUATION")
print("="*70)
for i, test_case in enumerate(test_cases, 1):
    query = test_case['query']
    print(f"\n{i}. Query: '{query}'")
    print("-" * 70)
    results = search_quotes(query, top_k=5)
    avg_similarity = np.mean([r['similarity'] for r in results])
    theme_matches = 0
    for result in results:
        text_lower = (result['quote'] + ' ' + result['author']).lower()
        for theme in test_case['expected_themes']:
            if theme in text_lower:
                theme_matches += 1
                break
    theme_coverage = theme_matches / len(results)
    author_matches = 0
    for result in results:
        author_lower = result['author'].lower()
        for expected_author in test_case['expected_authors']:
            if expected_author in author_lower:
                author_matches += 1
                break
    author_relevance = author_matches / len(results) if test_case['expected_authors'] else 0.5
    unique_authors = len(set(r['author'] for r in results))
    diversity_score = unique_authors / len(results)
    top_confidence = results[0]['similarity'] if results else 0
    retrieval_score = avg_similarity
    precision_score = theme_coverage
    relevance_score = (theme_coverage + author_relevance) / 2
    overall_score = (retrieval_score + precision_score + relevance_score + diversity_score) / 4
    eval_result = {
        'query': query,
        'description': test_case['description'],
        'avg_similarity': avg_similarity,
        'theme_coverage': theme_coverage,
        'author_relevance': author_relevance,
        'diversity_score': diversity_score,
        'top_confidence': top_confidence,
        'retrieval_score': retrieval_score,
        'precision_score': precision_score,
        'relevance_score': relevance_score,
        'overall_score': overall_score,
        'num_results': len(results)
    }
    evaluation_results.append(eval_result)
    print(f"  Average Similarity: {avg_similarity:.4f}")
    print(f"  Theme Coverage: {theme_coverage:.2%}")
    print(f"  Author Relevance: {author_relevance:.2%}")
    print(f"  Diversity Score: {diversity_score:.2%}")
    print(f"  Overall Score: {overall_score:.4f}")
    print(f"\n  Top Result:")
    top = results[0]
    print(f"    \"{top['quote'][:80]}...\"")
    print(f"    - {top['author']} (Similarity: {top['similarity']:.2%})")
print("\n" + "="*70)
print(" " * 20 + "AGGREGATE RESULTS")
print("="*70)
eval_df = pd.DataFrame(evaluation_results)
print("\n📊 Overall Performance:")
print("-" * 70)
print(f"Average Similarity Score:     {eval_df['avg_similarity'].mean():.4f}")
print(f"Average Theme Coverage:       {eval_df['theme_coverage'].mean():.2%}")
print(f"Average Author Relevance:     {eval_df['author_relevance'].mean():.2%}")
print(f"Average Diversity:            {eval_df['diversity_score'].mean():.2%}")
print(f"Average Top Confidence:       {eval_df['top_confidence'].mean():.4f}")
print(f"\n🎯 Composite Scores:")
print("-" * 70)
print(f"Retrieval Quality:            {eval_df['retrieval_score'].mean():.4f}")
print(f"Precision:                    {eval_df['precision_score'].mean():.4f}")
print(f"Relevance:                    {eval_df['relevance_score'].mean():.4f}")
print(f"Overall System Score:         {eval_df['overall_score'].mean():.4f}")
eval_df.to_csv('comprehensive_evaluation_results.csv', index=False)
print("\n✓ Saved: comprehensive_evaluation_results.csv")
with open('rag_evaluation_report_final.md', 'w', encoding='utf-8') as f:
    f.write("# RAG System Evaluation Report\n\n")
    f.write("## Executive Summary\n\n")
    f.write(f"**System**: Quote Search Engine (TF-IDF + Cosine Similarity)\n")
    f.write(f"**Evaluation Date**: January 17, 2026\n")
    f.write(f"**Test Queries**: {len(test_cases)}\n")
    f.write(f"**Dataset Size**: {len(quotes)} quotes\n\n")
    f.write("### Key Metrics\n\n")
    f.write("| Metric | Score | Interpretation |\n")
    f.write("|--------|-------|----------------|\n")
    f.write(f"| Average Similarity | {eval_df['avg_similarity'].mean():.4f} | Retrieval confidence |\n")
    f.write(f"| Theme Coverage | {eval_df['theme_coverage'].mean():.2%} | Content relevance |\n")
    f.write(f"| Author Relevance | {eval_df['author_relevance'].mean():.2%} | Expert source quality |\n")
    f.write(f"| Result Diversity | {eval_df['diversity_score'].mean():.2%} | Coverage breadth |\n")
    f.write(f"| **Overall Score** | **{eval_df['overall_score'].mean():.4f}** | **System performance** |\n\n")
    f.write("## Methodology\n\n")
    f.write("### Evaluation Framework\n\n")
    f.write("This evaluation uses a multi-dimensional approach to assess RAG system quality:\n\n")
    f.write("#### 1. Retrieval Quality (Average Similarity)\n")
    f.write("- **Definition**: Mean cosine similarity between query and retrieved quotes\n")
    f.write("- **Range**: 0.0 to 1.0\n")
    f.write("- **Interpretation**: Higher values indicate better semantic matching\n\n")
    f.write("#### 2. Precision (Theme Coverage)\n")
    f.write("- **Definition**: Percentage of results containing expected themes/keywords\n")
    f.write("- **Range**: 0% to 100%\n")
    f.write("- **Interpretation**: Measures how many results are actually relevant\n\n")
    f.write("#### 3. Relevance (Author + Theme Match)\n")
    f.write("- **Definition**: Combined score of theme and author relevance\n")
    f.write("- **Range**: 0.0 to 1.0\n")
    f.write("- **Interpretation**: Assesses both content and source quality\n\n")
    f.write("#### 4. Diversity\n")
    f.write("- **Definition**: Ratio of unique authors in results\n")
    f.write("- **Range**: 0% to 100%\n")
    f.write("- **Interpretation**: Higher diversity indicates broader coverage\n\n")
    f.write("## Detailed Results\n\n")
    for idx, row in eval_df.iterrows():
        f.write(f"### Query {idx + 1}: \"{row['query']}\"\n\n")
        f.write(f"**Description**: {row['description']}\n\n")
        f.write("**Metrics**:\n\n")
        f.write(f"- Average Similarity: {row['avg_similarity']:.4f}\n")
        f.write(f"- Theme Coverage: {row['theme_coverage']:.2%}\n")
        f.write(f"- Author Relevance: {row['author_relevance']:.2%}\n")
        f.write(f"- Diversity Score: {row['diversity_score']:.2%}\n")
        f.write(f"- Overall Score: {row['overall_score']:.4f}\n\n")
        results = search_quotes(test_cases[idx]['query'], top_k=1)
        if results:
            f.write("**Top Result**:\n\n")
            f.write(f"> \"{results[0]['quote']}\"\n\n")
            f.write(f"— {results[0]['author']} (Similarity: {results[0]['similarity']:.2%})\n\n")
        f.write("---\n\n")
    f.write("## Analysis\n\n")
    f.write("### Strengths\n\n")
    avg_score = eval_df['overall_score'].mean()
    if avg_score >= 0.5:
        f.write("- ✅ Strong overall performance with good retrieval quality\n")
    if eval_df['theme_coverage'].mean() >= 0.7:
        f.write("- ✅ High precision - most results are topically relevant\n")
    if eval_df['diversity_score'].mean() >= 0.7:
        f.write("- ✅ Good diversity - results cover multiple perspectives\n")
    f.write("\n### Areas for Improvement\n\n")
    if eval_df['avg_similarity'].mean() < 0.5:
        f.write("- ⚠️ Similarity scores could be improved with neural embeddings\n")
    if eval_df['author_relevance'].mean() < 0.5:
        f.write("- ⚠️ Consider boosting results from domain-expert authors\n")
    f.write("- ⚠️ TF-IDF is keyword-based; neural embeddings would capture deeper semantics\n")
    f.write("\n### Recommendations\n\n")
    f.write("1. **Upgrade to Neural Embeddings**: Use Sentence Transformers or BERT for better semantic understanding\n")
    f.write("2. **Implement Re-ranking**: Add a second-stage ranker to improve top results\n")
    f.write("3. **Query Expansion**: Expand queries with synonyms to improve recall\n")
    f.write("4. **Fine-tuning**: Train embeddings specifically on quote domain\n")
    f.write("5. **Author Weighting**: Boost results from famous/relevant authors\n")
    f.write("6. **User Feedback**: Collect ratings to improve ranking over time\n\n")
    f.write("## Conclusion\n\n")
    f.write(f"The RAG system demonstrates **{'strong' if avg_score >= 0.5 else 'moderate'}** ")
    f.write(f"performance with an overall score of **{avg_score:.4f}**. ")
    f.write("The TF-IDF-based approach provides fast, explainable results that are suitable ")
    f.write("for this dataset size. For production deployment or larger datasets, upgrading ")
    f.write("to neural embeddings would significantly improve semantic understanding and retrieval quality.\n\n")
    f.write("## Technical Details\n\n")
    f.write("**Vectorization Method**: TF-IDF with 1,000 features (1-2 n-grams)\n")
    f.write(f"**Search Method**: Cosine similarity\n")
    f.write(f"**Top-K**: 5 results per query\n")
    f.write(f"**Dataset**: {len(quotes)} quotes from Hugging Face (Abirate/english_quotes)\n")
    f.write(f"**Evaluation Queries**: {len(test_cases)} diverse test cases\n\n")
    f.write("---\n\n")
    f.write("*Report generated automatically by evaluate_comprehensive.py*\n")
print("✓ Saved: rag_evaluation_report_final.md")
eval_summary = {
    'evaluation_date': '2026-01-17',
    'system': 'Quote Search Engine (TF-IDF)',
    'dataset_size': len(quotes),
    'num_test_cases': len(test_cases),
    'metrics': {
        'avg_similarity': float(eval_df['avg_similarity'].mean()),
        'theme_coverage': float(eval_df['theme_coverage'].mean()),
        'author_relevance': float(eval_df['author_relevance'].mean()),
        'diversity_score': float(eval_df['diversity_score'].mean()),
        'overall_score': float(eval_df['overall_score'].mean())
    },
    'individual_results': evaluation_results
}
with open('evaluation_summary.json', 'w') as f:
    json.dump(eval_summary, f, indent=2)
print("✓ Saved: evaluation_summary.json")
print("\n" + "="*70)
print(" " * 15 + "✅ COMPREHENSIVE EVALUATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. comprehensive_evaluation_results.csv - Detailed metrics")
print("  2. rag_evaluation_report_final.md - Full evaluation report")
print("  3. evaluation_summary.json - Machine-readable summary")
print("\n" + "="*70)
