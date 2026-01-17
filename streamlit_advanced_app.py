import streamlit as st
import pandas as pd
import pickle
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
try:
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        groq_client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        )
        GROQ_AVAILABLE = True
    else:
        GROQ_AVAILABLE = False
except:
    GROQ_AVAILABLE = False
st.set_page_config(
    page_title="AI Quote Search Engine",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_resource
def load_search_system():
    try:
        with open('search_system.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("❌ Search system not found! Run: `python create_simple_search.py` first")
        st.stop()
data = load_search_system()
vectorizer = data['vectorizer']
tfidf_matrix = data['tfidf_matrix']
quotes = data['quotes']
authors = data['authors']
tags = data['tags']
combined_texts = data['combined_text']

def parse_query(query_text):
    """
    Robust query parser that extracts filters from natural language queries.
    Processing order: Century -> Tags -> Gender -> Author -> Topic cleanup
    """
    original_query = query_text
    filters = {
        'topic': query_text,
        'author': None,
        'gender': None,
        'tags': [],
        'century': None
    }
    
    # Track what we've removed to clean up the topic later
    removed_parts = []
    
    # 1. CENTURY FILTER - Check first to avoid capturing as author
    century_patterns = [
        (r'\b20th[- ]century\b', 20),
        (r'\b21st[- ]century\b', 21),
        (r'\b19th[- ]century\b', 19),
        (r'\b18th[- ]century\b', 18),
        (r'\b17th[- ]century\b', 17),
        (r'\bancient\b', 'ancient'),
        (r'\bmodern\b', 'modern'),
        (r'\bcontemporary\b', 'modern')
    ]
    for pattern, century in century_patterns:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match:
            filters['century'] = century
            removed_parts.append(match.group(0))
            query_text = re.sub(pattern, '', query_text, flags=re.IGNORECASE).strip()
            break
    
    # 2. TAG FILTER - Extract tags with quotes or after "tagged with"
    tag_patterns = [
        r"tagged?\s+with\s+(?:both\s+)?['\"]([^'\"]+)['\"](?:\s+and\s+['\"]([^'\"]+)['\"])?",
        r"about\s+['\"]([^'\"]+)['\"](?:\s+and\s+['\"]([^'\"]+)['\"])?",
        r"tags?:\s*['\"]?([a-zA-Z,\s-]+?)['\"]?(?:\s+and\s+['\"]([^'\"]+)['\"])?(?:\s|$)",
    ]
    for pattern in tag_patterns:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match:
            removed_parts.append(match.group(0))
            if match.lastindex >= 2 and match.group(2):
                filters['tags'] = [match.group(1).strip(), match.group(2).strip()]
            elif match.lastindex >= 1:
                tags_str = match.group(1)
                filters['tags'] = [t.strip() for t in tags_str.split(',') if t.strip()]
            query_text = re.sub(pattern, '', query_text, flags=re.IGNORECASE).strip()
            break
    
    # 3. GENDER FILTER - Check before author to avoid "women" being captured as author
    gender_keywords = {
        'women': [r'\bwomen\b', r'\bfemale\b', r'\bwoman\b'],
        'men': [r'\bmen\b', r'\bmale\b', r'\bman\b']
    }
    for gender, patterns in gender_keywords.items():
        for pattern in patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                filters['gender'] = gender
                removed_parts.append(match.group(0))
                query_text = re.sub(pattern, '', query_text, flags=re.IGNORECASE).strip()
                break
        if filters['gender']:
            break
    
    # 4. AUTHOR FILTER - Extract author name after "by" or "from"
    # Use non-greedy matching and stop at common keywords
    author_patterns = [
        r'by\s+([A-Z][a-zA-Z\s\.]+?)(?=\s+(?:about|on|tagged|from|$)|$)',
        r'from\s+([A-Z][a-zA-Z\s\.]+?)(?=\s+(?:about|on|tagged|by|$)|$)',
        r'written\s+by\s+([A-Z][a-zA-Z\s\.]+?)(?=\s+(?:about|on|tagged|$)|$)',
        r'author\s+([A-Z][a-zA-Z\s\.]+?)(?=\s+(?:about|on|tagged|$)|$)',
    ]
    
    for pattern in author_patterns:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match:
            author_name = match.group(1).strip()
            # Validate it's not a common word or filter keyword
            invalid_authors = ['authors', 'philosophers', 'writers', 'poets', 'people', 'quotes']
            if author_name.lower() not in invalid_authors and len(author_name) > 2:
                filters['author'] = author_name
                removed_parts.append(match.group(0))
                query_text = re.sub(pattern, '', query_text, flags=re.IGNORECASE).strip()
                break
    
    # 5. CLEAN UP TOPIC - Remove common filler words and normalize
    topic = original_query
    
    # Remove all identified filter parts
    for part in removed_parts:
        topic = re.sub(re.escape(part), '', topic, flags=re.IGNORECASE)
    
    # Remove common query prefixes
    topic = re.sub(r'^\s*(show\s+me|find|get|give\s+me|display|search\s+for|look\s+for)\s+', '', topic, flags=re.IGNORECASE)
    
    # Remove prepositions and articles left over from filter removal
    topic = re.sub(r'\s+(by|from|about|on|with|for|the|a|an)\s+', ' ', topic, flags=re.IGNORECASE)
    
    # Remove "authors", "quotes" suffixes
    topic = re.sub(r'\s+(authors?|quotes?|philosophers?|writers?|poets?)\s*$', '', topic, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    topic = re.sub(r'\s+', ' ', topic).strip()
    
    # Remove quotes if the entire topic is quoted
    topic = re.sub(r'^["\'](.+)["\']$', r'\1', topic)
    
    filters['topic'] = topic if topic else original_query
    
    return filters
def search_quotes(query, top_k=5, author_filter=None, gender_filter=None, tag_filters=None, century_filter=None):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    results_with_indices = []
    for idx, score in enumerate(similarities):
        results_with_indices.append((idx, score))
    if author_filter or gender_filter or tag_filters or century_filter:
        filtered_results = []
        women_authors = [
            'maya angelou', 'jane austen', 'emily dickinson', 'virginia woolf',
            'sylvia plath', 'toni morrison', 'margaret atwood', 'j.k. rowling',
            'agatha christie', 'harper lee', 'ayn rand', 'anne frank',
            'eleanor roosevelt', 'mother teresa', 'marilyn monroe', 'oprah winfrey',
            'helen keller', 'malala yousafzai', 'coco chanel', 'frida kahlo',
            'simone de beauvoir', 'george eliot', 'dorothy parker', 'anais nin',
            'edith wharton', 'zora neale hurston', 'bell hooks', 'audre lorde'
        ]
        modern_authors = [
            'maya angelou', 'jimi hendrix', 'marilyn monroe', 'martin luther king',
            'john f. kennedy', 'winston churchill', 'albert einstein', 'steve jobs',
            'nelson mandela', 'muhammad ali', 'bob marley', 'malcolm x',
            'mother teresa', 'dalai lama', 'oprah winfrey', 'stephen hawking'
        ]
        for idx, score in results_with_indices:
            author_lower = authors[idx].lower()
            if author_filter and author_filter.lower() not in author_lower:
                continue
            if gender_filter == 'women':
                is_woman = any(woman in author_lower for woman in women_authors)
                women_names = ['mary', 'elizabeth', 'margaret', 'helen', 'susan', 'sarah',
                              'emma', 'anna', 'alice', 'marie', 'julia', 'charlotte']
                if not is_woman:
                    is_woman = any(name in author_lower.split() for name in women_names)
                if not is_woman:
                    continue
            elif gender_filter == 'men':
                is_woman = any(woman in author_lower for woman in women_authors)
                if is_woman:
                    continue
            if tag_filters:
                quote_tags_str = str(tags[idx]).lower()
                if not all(tag.lower() in quote_tags_str for tag in tag_filters):
                    continue
            if century_filter:
                if century_filter == 20 or century_filter == 'modern':
                    if not any(modern_author in author_lower for modern_author in modern_authors):
                        continue
                elif century_filter == 'ancient':
                    ancient_keywords = ['plato', 'socrates', 'aristotle', 'confucius', 'buddha',
                                      'seneca', 'marcus aurelius', 'epictetus', 'cicero']
                    if not any(ancient in author_lower for ancient in ancient_keywords):
                        continue
            filtered_results.append((idx, score))
        results_with_indices = filtered_results
    results_with_indices.sort(key=lambda x: x[1], reverse=True)
    top_results = results_with_indices[:top_k]
    results = []
    for idx, score in top_results:
        results.append({
            'quote': quotes[idx],
            'author': authors[idx],
            'tags': tags[idx],
            'similarity': float(score),
            'combined_text': combined_texts[idx]
        })
    return results
def generate_summary(results, query):
    if not results:
        return "No relevant quotes found."
    num_results = len(results)
    avg_score = sum(r['similarity'] for r in results) / num_results
    unique_authors = len(set(r['author'] for r in results))
    summary = f"Found {num_results} relevant quotes for '{query}' "
    summary += f"with an average relevance score of {avg_score:.2%}. "
    summary += f"Results include quotes from {unique_authors} different author(s). "
    top = results[0]
    summary += f"The most relevant quote is from {top['author']} "
    summary += f"with a similarity score of {top['similarity']:.2%}."
    return summary
def create_json_response(results, query, filters):
    response = {
        'query': query,
        'filters': filters,
        'timestamp': datetime.now().isoformat(),
        'num_results': len(results),
        'summary': generate_summary(results, query),
        'results': []
    }
    for i, result in enumerate(results, 1):
        response['results'].append({
            'rank': i,
            'quote': result['quote'],
            'author': result['author'],
            'tags': eval(result['tags']) if isinstance(result['tags'], str) else result['tags'],
            'similarity_score': result['similarity'],
            'relevance_percentage': f"{result['similarity']:.2%}"
        })
    return response
def create_visualizations(results, all_quotes_data):
    viz_data = {
        'results': results,
        'all_quotes': all_quotes_data
    }
    return viz_data
def plot_similarity_distribution(results):
    if not results:
        return None
    scores = [r['similarity'] for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"#{i+1}" for i in range(len(scores))],
        y=scores,
        marker=dict(
            color=scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Similarity")
        ),
        text=[f"{s:.2%}" for s in scores],
        textposition='auto'
    ))
    fig.update_layout(
        title="Similarity Score Distribution",
        xaxis_title="Result Rank",
        yaxis_title="Similarity Score",
        height=400,
        template="plotly_white"
    )
    return fig
def plot_author_distribution(results):
    if not results:
        return None
    author_counts = Counter([r['author'] for r in results])
    fig = go.Figure(data=[
        go.Pie(
            labels=list(author_counts.keys()),
            values=list(author_counts.values()),
            hole=0.3,
            textinfo='label+percent'
        )
    ])
    fig.update_layout(
        title="Author Distribution in Results",
        height=400,
        template="plotly_white"
    )
    return fig
def plot_tag_distribution(results):
    if not results:
        return None
    all_tags = []
    for r in results:
        try:
            tag_list = eval(r['tags']) if isinstance(r['tags'], str) else r['tags']
            if isinstance(tag_list, list):
                all_tags.extend(tag_list)
        except:
            pass
    if not all_tags:
        return None
    tag_counts = Counter(all_tags)
    top_tags = dict(tag_counts.most_common(10))
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_tags.values()),
            y=list(top_tags.keys()),
            orientation='h',
            marker=dict(color='lightblue')
        )
    ])
    fig.update_layout(
        title="Top 10 Tags in Results",
        xaxis_title="Frequency",
        yaxis_title="Tags",
        height=400,
        template="plotly_white"
    )
    return fig
def plot_dataset_overview():
    unique_authors = len(set(authors))
    all_dataset_tags = []
    for tag in tags:
        try:
            tag_list = eval(tag) if isinstance(tag, str) else tag
            if isinstance(tag_list, list):
                all_dataset_tags.extend(tag_list)
        except:
            pass
    tag_counts = Counter(all_dataset_tags)
    top_tags = dict(tag_counts.most_common(15))
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_tags.keys()),
            y=list(top_tags.values()),
            marker=dict(
                color=list(top_tags.values()),
                colorscale='Blues',
                showscale=False
            )
        )
    ])
    fig.update_layout(
        title=f"Dataset Overview: Top 15 Tags (from {len(quotes)} quotes, {unique_authors} authors)",
        xaxis_title="Tags",
        yaxis_title="Frequency",
        height=500,
        template="plotly_white",
        xaxis_tickangle=-45
    )
    return fig
def generate_ai_summary(query, results):
    if not GROQ_AVAILABLE:
        return None
    context = "Retrieved quotes:\n\n"
    for i, r in enumerate(results[:3], 1):
        context += f'{i}. "{r["quote"]}" — {r["author"]}\n'
        if r.get('tags'):
            tags_list = r['tags']
            if isinstance(tags_list, str):
                try:
                    tags_list = eval(tags_list)
                except:
                    tags_list = []
            if isinstance(tags_list, list) and tags_list:
                context += f"   Tags: {', '.join(tags_list[:3])}\n"
        context += f"   Relevance: {r['similarity']:.1%}\n\n"
    
    prompt = f"User query: '{query}'\n\n{context}\nProvide a brief, insightful analysis of these quotes in relation to the query. Highlight common themes and key takeaways."
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a wise and insightful assistant that analyzes quotes and provides meaningful interpretations. Be concise, engaging, and thoughtful."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None
st.markdown('<div class="main-header">💬 AI Quote Search Engine</div>', unsafe_allow_html=True)
st.markdown("### Natural Language Search with Advanced Filtering")
st.markdown("---")
with st.sidebar:
    st.title("About")
    st.markdown("""
    This AI-powered search engine uses TF-IDF vectorization to find relevant quotes from a database of 2,508 quotes. 
    Use natural language queries with advanced filters!
    """)
    
    st.title("Example Queries")
    st.markdown("""
    - *"quotes about courage by women authors"*
    - *"wisdom from ancient philosophers"*
    - *"quotes tagged with 'life' and 'love'"*
    - *"inspirational quotes by Shakespeare"*
    - *"quotes about success and failure"*
    """)
    
    st.title("Statistics")
    st.metric("Total Quotes", "2,508")
    st.metric("Unique Authors", f"{len(set(authors))}")
    st.metric("Search Speed", "<100ms")
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., Show me quotes about courage by women authors...",
        help="Try natural language! You can filter by author, gender, or topic."
    )
with col2:
    top_k = st.slider("📊 Results:", 1, 20, 5, help="Number of quotes to retrieve")
with st.expander("🎛️ Advanced Options"):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        show_json = st.checkbox("Show JSON Response", value=False)
    with col_b:
        show_scores = st.checkbox("Show Similarity Scores", value=True)
    with col_c:
        show_tags = st.checkbox("Show Tags", value=True)
    with col_d:
        show_viz = st.checkbox("Show Visualizations", value=False)
    if GROQ_AVAILABLE:
        st.markdown("---")
        use_ai_summary = st.checkbox("🤖 AI-Powered Summary (Groq Llama 3.1)", value=True,
                                      help="Generate natural language insights using AI")
    else:
        use_ai_summary = False
    if GROQ_AVAILABLE:
        st.markdown("---")
        use_ai_summary = st.checkbox("🤖 AI-Powered Summary (Groq Llama 3.1)", value=True,
                                      key="ai_summary_toggle",
                                      help="Generate natural language insights using AI")
    else:
        use_ai_summary = False
if st.button("🔍 Search", type="primary", use_container_width=True) or query:
    if query:
        with st.spinner("🔎 Searching through 2,508 quotes..."):
            filters = parse_query(query)
            filter_parts = []
            if filters['topic']:
                filter_parts.append(f"Topic: '{filters['topic']}'")
            if filters['author']:
                filter_parts.append(f"Author: '{filters['author']}'")
            if filters['gender']:
                filter_parts.append(f"Gender: '{filters['gender']}'")
            if filters['tags']:
                filter_parts.append(f"Tags: {filters['tags']}")
            if filters['century']:
                filter_parts.append(f"Century: '{filters['century']}'")
            if len(filter_parts) > 1:
                st.info(f"🎯 **Detected Multi-hop Filters**: {' | '.join(filter_parts)}")
            elif filters['author'] or filters['gender'] or filters['tags'] or filters['century']:
                st.info(f"🎯 **Detected Filters**: {' | '.join(filter_parts)}")
            results = search_quotes(
                filters['topic'],
                top_k=top_k,
                author_filter=filters['author'],
                gender_filter=filters['gender'],
                tag_filters=filters['tags'],
                century_filter=filters['century']
            )
            if not results:
                st.warning(f"⚠️ No results found for '{query}'. Try broadening your search or removing filters.")
            else:
                json_response = create_json_response(results, query, filters)
                st.success(f"✅ {json_response['summary']}")
                if use_ai_summary and GROQ_AVAILABLE:
                    with st.spinner("🤖 Generating AI insights..."):
                        ai_summary = generate_ai_summary(query, results)
                        if ai_summary:
                            st.markdown("---")
                            st.markdown("### 🤖 AI-Powered Insights")
                            st.info(ai_summary)
                if show_json:
                    st.markdown("---")
                    st.subheader("📋 Structured JSON Response")
                    st.json(json_response)
                    json_str = json.dumps(json_response, indent=2)
                    st.download_button(
                        label="⬇️ Download JSON Results",
                        data=json_str,
                        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                if show_viz and results:
                    st.markdown("---")
                    st.subheader("📊 Data Visualizations")
                    viz_tabs = st.tabs(["Similarity Scores", "Author Distribution", "Tag Distribution", "Dataset Overview"])
                    with viz_tabs[0]:
                        st.plotly_chart(plot_similarity_distribution(results), use_container_width=True)
                    with viz_tabs[1]:
                        fig = plot_author_distribution(results)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough data for author distribution")
                    with viz_tabs[2]:
                        fig = plot_tag_distribution(results)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No tags found in results")
                    with viz_tabs[3]:
                        st.plotly_chart(plot_dataset_overview(), use_container_width=True)
                st.markdown("---")
                st.subheader(f"📊 Top {len(results)} Results")
                for i, result in enumerate(results, 1):
                    with st.container():
                        # Header row with rank and similarity
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.markdown(f"### #{i}")
                        with cols[1]:
                            if show_scores:
                                score_color = "🟢" if result['similarity'] > 0.7 else "🟡" if result['similarity'] > 0.4 else "🟠"
                                st.markdown(f"**Similarity:** {score_color} {result['similarity']:.2%}")
                        
                        # Quote text - prominent display
                        st.markdown(f"### 💬 \"{result['quote']}\"")
                        
                        # Author name
                        st.markdown(f"**— {result['author']}**")
                        
                        # Tags row
                        if show_tags and result['tags']:
                            try:
                                tag_list = eval(result['tags']) if isinstance(result['tags'], str) else result['tags']
                                if isinstance(tag_list, list) and tag_list:
                                    tag_str = " ".join([f"`{tag}`" for tag in tag_list[:5]])
                                    st.markdown(f"**Tags:** {tag_str}")
                            except:
                                pass
                        
                        # Action buttons
                        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
                        with col_btn1:
                            if st.button(f"📋 Copy", key=f"copy_{i}"):
                                st.code(f'"{result["quote"]}" - {result["author"]}', language="text")
                        
                        st.markdown("---")
                st.subheader("📥 Export Results")
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    json_str = json.dumps(json_response, indent=2)
                    st.download_button(
                        label="📄 Export as JSON",
                        data=json_str,
                        file_name=f"quotes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                with col_exp2:
                    txt_content = f"Search Query: {query}\n"
                    txt_content += f"Timestamp: {json_response['timestamp']}\n"
                    txt_content += f"Results: {len(results)}\n\n"
                    txt_content += "="*70 + "\n\n"
                    for i, result in enumerate(results, 1):
                        txt_content += f"{i}. \"{result['quote']}\"\n"
                        txt_content += f"   — {result['author']}\n"
                        txt_content += f"   Similarity: {result['similarity']:.2%}\n\n"
                    st.download_button(
                        label="📝 Export as TXT",
                        data=txt_content,
                        file_name=f"quotes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
    else:
        st.warning("⚠️ Please enter a search query")
st.markdown("---")
