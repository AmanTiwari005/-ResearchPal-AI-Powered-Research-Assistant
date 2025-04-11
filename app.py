from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import requests
import re
from groq import Groq
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import PyPDF2
import io
import torch
from collections import Counter
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("WordCloud library not available. Word cloud visualizations will be skipped.")
from textblob import TextBlob
from gtts import gTTS
from io import BytesIO
import json
import pickle
import uuid
from urllib.parse import quote
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import nbformat as nbf
import huggingface_hub

# Initialize GROQ chat model
def init_groq_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

# Load Grok model
if "llm_groq" not in st.session_state:
    try:
        st.session_state["llm_groq"] = init_groq_model()
    except ValueError as e:
        st.error(str(e))
        st.stop()

llm_groq = st.session_state["llm_groq"]

# Check if GPU is available (will be CPU on Streamlit Cloud)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"**Using device:** {device}")

# Load models with error handling and caching
st.write("Initializing models...")
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
except Exception as e:
    st.error(f"Failed to load summarizer model: {e}")
    st.stop()

try:
    with st.spinner("Loading SentenceTransformer model (this may take a moment)..."):
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device, cache_folder="./model_cache")
    st.success("SentenceTransformer model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load SentenceTransformer model: {e}")
    st.info("Ensure you have an internet connection and sufficient disk space. The app will exit.")
    st.stop()

# Cache translator model
@st.cache_resource
def load_translator(model_name):
    try:
        return pipeline("translation", model=model_name, device=-1)
    except Exception as e:
        st.error(f"Failed to load translator model: {e}")
        return None

translator = load_translator("Helsinki-NLP/opus-mt-en-es")
if translator is None:
    st.warning("Translation feature will be unavailable.")

# Utility Functions (Existing)
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        return text
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""

def split_into_paragraphs(text):
    cleaned_text = re.sub(r'\s*\n\s*', '\n', text.strip())
    cleaned_text = re.sub(r'(\.\s*\n)', '.\n\n', cleaned_text)
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip() and len(p.strip()) > 10]
    return paragraphs if paragraphs else [text[:1000]]

def search_keywords(paper_text, query):
    if not query or not paper_text:
        return []
    paragraphs = split_into_paragraphs(paper_text)
    matches = []
    query = query.strip()
    query_words = query.split()
    pattern = r'\b' + re.escape(query) + r'\b'
    for i, paragraph in enumerate(paragraphs):
        if re.search(pattern, paragraph, re.IGNORECASE):
            matches.append((i, paragraph))
        elif len(query_words) > 1:
            if all(re.search(r'\b' + re.escape(word) + r'\b', paragraph, re.IGNORECASE) for word in query_words):
                matches.append((i, paragraph))
    return matches

def highlight_keywords(text, query, context_size=500):
    match = re.search(r'\b' + re.escape(query) + r'\b', text, re.IGNORECASE)
    if match:
        start = max(0, match.start() - context_size)
        end = min(len(text), match.end() + context_size)
        snippet = text[start:end]
        highlighted = re.sub(
            r'\b' + re.escape(query) + r'\b',
            r"**\g<0>**",
            snippet,
            flags=re.IGNORECASE
        )
        return "..." + highlighted + "..." if start > 0 or end < len(text) else highlighted
    return text

def count_keyword_frequency(paper_text, query):
    pattern = r'\b' + re.escape(query) + r'\b'
    return len(re.findall(pattern, paper_text, re.IGNORECASE))

@st.cache_data
def suggest_similar_terms(query):
    try:
        response = llm_groq.invoke(
            f"Given the keyword '{query}', suggest 5 related terms for a research paper context. Return each term on a new line."
        )
        terms = response.content.strip().split('\n')[:5]
        valid_terms = [term.strip() for term in terms if term.strip()]
        if not valid_terms:
            raise ValueError("No valid terms returned.")
        return valid_terms
    except Exception as e:
        st.warning(f"Failed to fetch similar terms: {e}. Using fallback.")
        fallback = {
            "machine learning": ["deep learning", "neural networks", "artificial intelligence", "data mining", "pattern recognition"],
            "data": ["big data", "data analysis", "statistics", "information", "database"],
        }
        return fallback.get(query.lower(), ["term1", "term2", "term3", "term4", "term5"])[:5]

def extract_metadata_identifiers(paper_text):
    doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
    doi_match = re.search(doi_pattern, paper_text)
    doi = doi_match.group(0) if doi_match else None
    lines = paper_text.split("\n")
    title = next((line.strip() for line in lines if len(line.strip()) > 10 and line.strip()[0].isupper()), "Untitled")
    return doi, title

@st.cache_data
def get_citation_from_apis(paper_text, style="APA"):
    doi, title = extract_metadata_identifiers(paper_text)
    metadata = {}
    if doi:
        url = f"https://api.crossref.org/works/{doi}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()["message"]
                metadata["authors"] = data.get("author", [{"family": "Unknown", "given": ""}])
                metadata["title"] = data.get("title", ["Untitled"])[0]
                metadata["year"] = data.get("issued", {}).get("date-parts", [[2023]])[0][0]
        except Exception as e:
            st.warning(f"CrossRef failed: {e}")
    if not metadata:
        base_url = "https://api.openalex.org/works"
        query = f"doi:{doi}" if doi else f"filter=title.search:{title}"
        url = f"{base_url}?{query}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()["results"][0] if "results" in response.json() and response.json()["results"] else {}
                if data:
                    metadata["authors"] = [{"family": a["author"]["display_name"].split()[-1], "given": " ".join(a["author"]["display_name"].split()[:-1])} for a in data.get("authorships", [{"author": {"display_name": "Unknown"}}])]
                    metadata["title"] = data.get("title", title)
                    metadata["year"] = data.get("publication_year", 2023)
        except Exception as e:
            st.warning(f"OpenAlex failed: {e}")
    if not metadata:
        base_url = "https://api.semanticscholar.org/graph/v1/paper"
        query = f"DOI:{doi}" if doi else f"search?query={title}&limit=1"
        url = f"{base_url}/{query}?fields=title,authors,year"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()["data"][0] if "data" in response.json() else response.json()
                metadata["authors"] = [{"family": a["name"].split()[-1], "given": " ".join(a["name"].split()[:-1])} for a in data.get("authors", [{"name": "Unknown"}])]
                metadata["title"] = data.get("title", title)
                metadata["year"] = data.get("year", 2023)
        except Exception as e:
            st.warning(f"Semantic Scholar failed: {e}")
    if not metadata:
        response = llm_groq.invoke(
            f"Extract metadata (title, authors as a comma-separated list, year) from this paper and format it as an {style} citation. Return only the citation: {paper_text[:4000]}"
        )
        return response.content.strip()
    authors = metadata["authors"]
    title = metadata["title"]
    year = metadata["year"]
    if style == "APA":
        authors_str = ", ".join([f"{a['family']}, {a.get('given', '')[:1]}." for a in authors])
        return f"{authors_str}. ({year}). {title}."
    elif style == "MLA":
        authors_str = " and ".join([f"{a.get('given', '')} {a['family']}" for a in authors])
        return f"{authors_str}. \"{title}.\" {year}."
    elif style == "IEEE":
        authors_str = ", ".join([f"{a.get('given', '')[:1]}. {a['family']}" for a in authors])
        return f"{authors_str}, \"{title},\" {year}."

@st.cache_data
def suggest_references(paper_text, offset=0, limit=5):
    doi, title = extract_metadata_identifiers(paper_text)
    suggestions = []
    if title:
        url = f"https://api.openalex.org/works?filter=title.search:{title}&per-page={limit}&page={offset + 1}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    for result in data["results"]:
                        authors = ", ".join([a["author"]["display_name"] for a in result.get("authorships", [])])
                        ref_title = result.get("title", "Untitled")
                        year = result.get("publication_year", 2023)
                        suggestions.append(f"{authors}. ({year}). {ref_title}.")
        except requests.RequestException as e:
            st.warning(f"OpenAlex reference suggestion failed: {e}")
    if len(suggestions) < limit:
        with st.spinner("Generating additional suggestions..."):
            prompt = (
                f"Based on this paper excerpt: {paper_text[:2000]}, suggest {limit - len(suggestions)} "
                f"related references in APA style. Return each as a single line."
            )
            response = llm_groq.invoke(prompt)
            ai_suggestions = response.content.strip().split("\n")
            suggestions.extend(ai_suggestions[:limit - len(suggestions)])
    return suggestions[:limit]

def cross_reference_citations(paper_text):
    citation_pattern = r"\[\d+\]|\(\w+,\s*\d{4}\)"
    matches = re.findall(citation_pattern, paper_text)
    if matches:
        return [f"Found citation: {match}" for match in matches[:5]]
    return ["No clear cross-references identified."]

def chunk_text(text, max_chunk_length=512):
    words = text.split()
    for i in range(0, len(words), max_chunk_length):
        yield ' '.join(words[i:i + max_chunk_length])

def summarize_text(text, min_chunk_length=200):
    summary = ""
    for chunk in chunk_text(text):
        chunk = chunk.strip()
        if len(chunk) == 0 or len(chunk.split()) < 5:
            continue
        try:
            response = llm_groq.invoke(
                f"Provide a detailed summary of this text (at least {min_chunk_length} characters): {chunk}"
            )
            summary += response.content.strip() + " "
        except Exception as e:
            st.error(f"Error summarizing chunk: {e}")
            return None
    return summary.strip()

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def text_to_speech_gtts(text, sentiment_adjustment=False):
    try:
        if sentiment_adjustment:
            sentiment = analyze_sentiment(text)
            speech_rate = 'slow' if sentiment < 0 else 'normal'
        else:
            speech_rate = 'normal'
        tts = gTTS(text=text[:2000], lang='en', slow=(speech_rate == 'slow'))
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def enhance_podcast_text(text):
    greeting = "Hello, dear listeners! Welcome to this special podcast where we dive into an exciting research paper. Let’s explore its key insights together."
    middle = "Now, let’s take a moment to appreciate the depth of this work as we move into more fascinating details."
    closing = "That’s all for today’s episode. Thank you for joining me on this journey through the research. Stay curious, and until next time, goodbye!"
    words = text.split()
    mid_point = len(words) // 2
    first_half = " ".join(words[:mid_point])
    second_half = " ".join(words[mid_point:])
    return f"{greeting} {first_half} {middle} {second_half} {closing}"

# New Utility Functions
@st.cache_data
def fetch_related_papers(query, max_results=5):
    try:
        url = f"http://export.arxiv.org/api/query?search_query={quote(query)}&max_results={max_results}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            from xml.etree import ElementTree
            root = ElementTree.fromstring(response.content)
            papers = []
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                title = entry.find("{http://www.w3.org/2005/Atom}title").text
                doi = entry.find("{http://www.w3.org/2005/Atom}id").text
                summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
                papers.append({"title": title, "doi": doi, "summary": summary[:200]})
            return papers
    except Exception as e:
        st.warning(f"Failed to fetch papers: {e}")
        return []

@st.cache_data
def build_citation_network(doi, max_nodes=10):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=title,references,citations"
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        G = nx.DiGraph()
        G.add_node(data["title"], type="root")
        for ref in data.get("references", [])[:max_nodes//2]:
            if "title" in ref:
                G.add_node(ref["title"], type="reference")
                G.add_edge(data["title"], ref["title"])
        for cit in data.get("citations", [])[:max_nodes//2]:
            if "title" in cit:
                G.add_node(cit["title"], type="citation")
                G.add_edge(cit["title"], data["title"])
        return G
    except Exception as e:
        st.warning(f"Failed to build network: {e}")
        return None

def create_notebook(code):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("Generated Code"))
    nb.cells.append(nbf.v4.new_code_cell(code))
    return nb

def save_cache(key, data):
    try:
        with open(f"cache_{key}.pkl", "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"Failed to save cache: {e}")

def load_cache(key):
    try:
        with open(f"cache_{key}.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def fetch_forum_questions(query):
    try:
        url = f"https://api.stackexchange.com/2.3/search?site=academia.stackexchange.com&intitle={quote(query)}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return [{"title": q["title"], "link": q["link"]} for q in response.json()["items"][:3]]
        return []
    except Exception as e:
        st.warning(f"Failed to fetch questions: {e}")
        return []

# Streamlit App
st.title("📚 Research Paper Management and Chatbot")
st.markdown("Welcome to your AI-powered research assistant! Explore, summarize, and interact with your research paper effortlessly.")

# Initialize session state
if "profile" not in st.session_state:
    st.session_state["profile"] = {"citation_style": "APA", "research_field": "", "keywords": []}
if "last_interaction" not in st.session_state:
    st.session_state["last_interaction"] = datetime.now()

# Sidebar
st.sidebar.title("⚙️ Navigation")
uploaded_file = st.sidebar.file_uploader("📤 Upload your research paper (PDF):", type="pdf")
if uploaded_file:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.sidebar.error("File too large. Upload a file under 10MB.")
    else:
        with st.spinner("Extracting text from PDF..."):
            paper_text = extract_text_from_pdf(uploaded_file)
            if paper_text:
                st.session_state["paper_text"] = paper_text
                save_cache("paper_text", paper_text)
                st.sidebar.success("✅ File uploaded successfully!")
            else:
                st.sidebar.error("Failed to extract text from PDF.")
else:
    st.sidebar.warning("⚠️ Please upload a research paper to proceed.")

# Researcher Profile
st.sidebar.subheader("Researcher Profile")
with st.sidebar.form("profile_form"):
    citation_style = st.selectbox("Preferred Citation Style:", ["APA", "MLA", "IEEE"], 
                                 index=["APA", "MLA", "IEEE"].index(st.session_state["profile"]["citation_style"]))
    research_field = st.text_input("Research Field:", st.session_state["profile"]["research_field"])
    keywords = st.text_input("Favorite Keywords (comma-separated):", 
                            ",".join(st.session_state["profile"]["keywords"]))
    if st.form_submit_button("Save Profile"):
        st.session_state["profile"] = {
            "citation_style": citation_style,
            "research_field": research_field,
            "keywords": [k.strip() for k in keywords.split(",") if k.strip()]
        }
        st.sidebar.success("Profile saved!")

# Smart Notifications
time_diff = datetime.now() - st.session_state["last_interaction"]
if time_diff > timedelta(days=7) and st.session_state.get("drafted_content"):
    st.sidebar.warning("Your draft hasn’t been updated in over a week!")
st.sidebar.subheader("Set Deadline")
deadline = st.sidebar.date_input("Submission Deadline:")
if deadline and deadline < datetime.now().date():
    st.sidebar.error("Deadline passed! Consider revising your schedule.")
st.session_state["last_interaction"] = datetime.now()

# Tutorial
st.sidebar.subheader("Get Started")
if st.sidebar.button("Start Tutorial"):
    st.session_state["tutorial_step"] = 0

if "tutorial_step" in st.session_state:
    steps = [
        ("Welcome!", "Upload a PDF to begin analyzing your paper."),
        ("Discover", "Use the Research Paper Discovery tool to search keywords."),
        ("Chat", "Ask questions about your paper with the Chatbot.")
    ]
    step = st.session_state["tutorial_step"]
    if step < len(steps):
        st.markdown(f"""
        <div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px;'>
        <h3>{steps[step][0]}</h3><p>{steps[step][1]}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Next"):
            st.session_state["tutorial_step"] += 1
    else:
        st.success("Tutorial completed!")
        del st.session_state["tutorial_step"]

# Mobile Optimization
st.markdown("""
<style>
.stTextInput > div > input {font-size: 16px;}
.stButton > button {width: 100%; padding: 10px;}
@media (max-width: 600px) {
    .stMarkdown {font-size: 14px;}
    .stSelectbox {margin-bottom: 10px;}
}
</style>
""", unsafe_allow_html=True)

options = st.sidebar.radio("Choose a feature:", [
    "🔍 Research Paper Discovery", "💬 Interactive Chatbot", "✍️ Writing Assistance",
    "📑 Citation and Reference Management", "🖥️ Data and Code Assistance", 
    "📊 Visualizations", "🎙️ Paper to Podcast", "📈 Dashboard"
])

if "paper_text" not in st.session_state:
    st.warning("⚠️ Please upload a research paper first.")
else:
    paper_text = st.session_state["paper_text"]

    # Research Paper Discovery
    if options == "🔍 Research Paper Discovery":
        st.header("🔍 Research Paper Discovery")
        col1, col2 = st.columns([1, 1])
        with col1:
            query = st.text_input("Enter keywords to search within your paper:", key="mobile_query")
        with col2:
            if st.button("Search", key="mobile_search"):
                with st.spinner("Searching within your paper..."):
                    matches = search_keywords(paper_text, query)
                    frequency = count_keyword_frequency(paper_text, query)
                    st.write(f"**Keyword '{query}' appears {frequency} times in the document.**")
                    if matches:
                        st.write(f"**Found {len(matches)} paragraph matches for '{query}':**")
                        for i, (para_num, paragraph) in enumerate(matches[:20]):
                            st.write(f"**Match {i + 1} (Paragraph {para_num + 1}):**")
                            st.markdown(highlight_keywords(paragraph, query), unsafe_allow_html=True)
                        if len(matches) > 20:
                            st.info(f"Showing first 20 of {len(matches)} matches.")
                    else:
                        st.warning(f"No paragraph matches found for '{query}'.")
                        st.write("**Sample text:**", paper_text[:500])
        with st.expander("Explore Similar Terms"):
            st.write("**Suggested related terms:**")
            similar_terms = suggest_similar_terms(query)
            for term in similar_terms:
                if st.button(term, key=f"term_{term}"):
                    st.session_state["query"] = term
                    st.experimental_rerun()

        st.subheader("Offline Mode")
        cached_text = load_cache("paper_text")
        if cached_text:
            offline_query = st.text_input("Search cached paper:")
            if offline_query:
                matches = search_keywords(cached_text, offline_query)
                if matches:
                    for i, (para_num, paragraph) in enumerate(matches[:5]):
                        st.write(f"**Match {i + 1} (Paragraph {para_num + 1}):**")
                        st.markdown(highlight_keywords(paragraph, offline_query))
                else:
                    st.info("No matches found in cached text.")

        with st.expander("Find Related Papers"):
            lit_query = st.text_input("Enter topic or keywords for literature search:")
            if lit_query and st.button("Search Literature"):
                with st.spinner("Fetching related papers..."):
                    papers = fetch_related_papers(lit_query)
                    if papers:
                        st.write("**Related Papers:**")
                        for paper in papers:
                            st.write(f"- **{paper['title']}** ([{paper['doi']}]({paper['doi']}))")
                            st.write(f"  {paper['summary']}...")
                    else:
                        st.info("No papers found.")

        st.subheader("Research Gap Analysis")
        if st.button("Identify Research Gaps"):
            with st.spinner("Analyzing for gaps..."):
                prompt = f"Based on this paper excerpt: {paper_text[:4000]}, identify 3-5 underexplored research areas or gaps. List each gap as a bullet point."
                response = llm_groq.invoke(prompt)
                st.write("**Research Gaps:**")
                st.markdown(response.content.strip())

        st.subheader("Share Paper Summary")
        if st.button("Generate Shareable Summary"):
            with st.spinner("Creating summary..."):
                summary = summarize_text(paper_text[:2000], min_chunk_length=100)
                if summary:
                    share_id = str(uuid.uuid4())
                    st.session_state[f"summary_{share_id}"] = summary
                    share_url = f"https://your-app-url/share?summary={share_id}"
                    st.write("**Shareable Link:**")
                    st.write(share_url)
                else:
                    st.error("Failed to generate summary.")

        st.subheader("Community Q&A")
        forum_query = st.text_input("Search for related discussions:")
        if forum_query and st.button("Search Forums"):
            with st.spinner("Fetching discussions..."):
                questions = fetch_forum_questions(forum_query)
                if questions:
                    st.write("**Related Discussions:**")
                    for q in questions:
                        st.write(f"- [{q['title']}]({q['link']})")
                else:
                    st.info("No discussions found.")

    # Interactive Chatbot
    elif options == "💬 Interactive Chatbot":
        st.header("💬 Interactive Research Paper Chatbot")
        context = st.session_state.get("context", "")
        user_input = st.text_input("💡 Ask a question about your paper:")
        if user_input:
            with st.spinner("Generating answer..."):
                response = llm_groq.invoke(
                    f"Context: {paper_text[:4000]}\nQuestion: {user_input}\nAnswer:"
                )
                answer = response.content.strip()
                st.write("**Answer:**", answer)
                context += f"\nQ: {user_input}\nA: {answer}"
                st.session_state["context"] = context
            with st.expander("📜 Conversation History"):
                st.write(context)

    # Writing Assistance
    elif options == "✍️ Writing Assistance":
        st.header("✍️ Writing Assistance")
        task = st.radio("Choose a task:", [
            "📝 Generate Outline", "✍️ Draft and Paraphrase", 
            "📄 Abstract Generator", "🔍 Style and Grammar Checker", 
            "🌐 Translate Text", "📑 Journal Formatter", 
            "🖋️ Peer Review Simulator", "📋 Paper Templates"
        ])

        if task == "📝 Generate Outline":
            use_uploaded_paper = st.toggle("Base outline on uploaded paper?", value=True)
            if use_uploaded_paper:
                topic = st.text_input("Enter a specific focus (optional):")
                if st.button("Generate Outline from Paper"):
                    with st.spinner("Generating outline..."):
                        prompt = f"Generate a research paper outline based on this content: {paper_text[:4000]}"
                        if topic:
                            prompt += f" with a focus on: {topic}"
                        response = llm_groq.invoke(prompt)
                        st.write("**Outline:**", response.content.strip())
            else:
                topic = st.text_input("Enter topic for new paper:")
                if topic and st.button("Generate Outline for New Paper"):
                    with st.spinner("Generating outline..."):
                        response = llm_groq.invoke(
                            f"Generate a detailed research paper outline for a new paper on the topic: {topic}"
                        )
                        st.write("**Outline:**", response.content.strip())

        elif task == "✍️ Draft and Paraphrase":
            st.subheader("✍️ Draft and Paraphrase")
            if "drafted_content" not in st.session_state:
                st.session_state["drafted_content"] = ""
            st.write("### Draft Section")
            section = st.text_input("Enter the section (e.g., Introduction):")
            if section and st.button("Generate Draft"):
                with st.spinner("Drafting section..."):
                    response = llm_groq.invoke(
                        f"Draft the {section} section based on this paper: {paper_text[:4000]}"
                    )
                    drafted_text = response.content.strip()
                    st.session_state["drafted_content"] = drafted_text
                    st.write("**Drafted Content:**")
                    st.write(drafted_text)
            if st.session_state["drafted_content"]:
                st.write("**Current Draft:**")
                st.write(st.session_state["drafted_content"])
            st.markdown("---")
            st.write("### Paraphrase Text")
            text_to_paraphrase = st.text_area(
                "Text to paraphrase:", value=st.session_state["drafted_content"]
            )
            if text_to_paraphrase and st.button("Paraphrase Text"):
                with st.spinner("Paraphrasing..."):
                    response = llm_groq.invoke(
                        f"Paraphrase the following text: {text_to_paraphrase}"
                    )
                    st.write("**Paraphrased Text:**")
                    st.write(response.content.strip())

        elif task == "📄 Abstract Generator":
            st.subheader("Abstract Generator")
            tone = st.selectbox("Abstract Tone:", ["Concise", "Detailed", "Balanced"])
            if st.button("Generate Abstract"):
                with st.spinner("Generating abstract..."):
                    prompt = f"Generate a {tone.lower()} abstract (150-250 words) for this paper: {paper_text[:4000]}"
                    response = llm_groq.invoke(prompt)
                    abstract = response.content.strip()
                    st.write("**Generated Abstract:**")
                    st.write(abstract)
                    st.download_button("Download Abstract", abstract, "abstract.txt", "text/plain")

        elif task == "🔍 Style and Grammar Checker":
            st.subheader("Style and Grammar Checker")
            input_text = st.text_area("Paste text to check:", value=st.session_state.get("drafted_content", ""))
            if input_text and st.button("Check Style and Grammar"):
                with st.spinner("Checking text..."):
                    blob = TextBlob(input_text)
                    grammar_issues = [s for s in blob.sentences if s.correct() != str(s)]
                    st.write("**Grammar Issues:**")
                    if grammar_issues:
                        for issue in grammar_issues[:5]:
                            st.write(f"- Original: {issue}")
                            st.write(f"  Suggested: {issue.correct()}")
                    else:
                        st.write("- No grammar issues found.")
                    prompt = f"Suggest improvements for academic style in this text, avoiding jargon: {input_text[:2000]}"
                    response = llm_groq.invoke(prompt)
                    st.write("**Style Suggestions:**")
                    st.write(response.content.strip())

        elif task == "🌐 Translate Text":
            st.subheader("Translate Text")
            languages = {"Spanish": "es", "French": "fr", "German": "de"}
            target_lang = st.selectbox("Target Language:", list(languages.keys()))
            trans_text = st.text_area("Text to translate:", value=st.session_state.get("drafted_content", ""))
            if trans_text and st.button("Translate"):
                with st.spinner("Translating..."):
                    if translator:
                        result = translator(trans_text[:500], max_length=512)[0]["translation_text"]
                        st.write("**Translated Text:**")
                        st.write(result)
                    else:
                        st.warning("Translation unavailable due to model loading failure.")

        elif task == "📑 Journal Formatter":
            st.subheader("Journal Formatter")
            template = st.selectbox("Journal Template:", ["IEEE", "Springer", "Generic"])
            if st.button("Format Paper"):
                with st.spinner("Formatting..."):
                    prompt = f"Format this paper excerpt as a {template} journal submission: {paper_text[:4000]}"
                    response = llm_groq.invoke(prompt)
                    formatted_text = response.content.strip()
                    st.write("**Formatted Paper:**")
                    st.write(formatted_text)
                    st.download_button("Download Formatted Paper", formatted_text, f"paper_{template}.txt", "text/plain")

        elif task == "🖋️ Peer Review Simulator":
            st.subheader("Peer Review Simulator")
            if st.button("Simulate Peer Review"):
                with st.spinner("Generating feedback..."):
                    prompt = f"Act as a peer reviewer and critique this paper for clarity, novelty, and methodology: {paper_text[:4000]}. Provide 3-5 specific suggestions."
                    response = llm_groq.invoke(prompt)
                    st.write("**Reviewer Feedback:**")
                    st.markdown(response.content.strip())

        elif task == "📋 Paper Templates":
            st.subheader("Paper Templates")
            templates = {
                "Review Paper": "1. Introduction\n2. Literature Review\n3. Synthesis\n4. Conclusion",
                "Empirical Study": "1. Introduction\n2. Methods\n3. Results\n4. Discussion"
            }
            template_choice = st.selectbox("Choose Template:", list(templates.keys()))
            if st.button("Load Template"):
                st.write("**Template Preview:**")
                st.write(templates[template_choice])
                if st.button("Use Template"):
                    st.session_state["drafted_content"] = templates[template_choice]
                    st.write("Template loaded into draft!")

    # Citation and Reference Management
    elif options == "📑 Citation and Reference Management":
        st.header("📑 Citation and Reference Management")
        st.subheader("Auto-Citation")
        citation_style = st.session_state["profile"]["citation_style"]
        if st.button("Generate Citation"):
            with st.spinner("Fetching citation..."):
                citation = get_citation_from_apis(paper_text, style=citation_style)
                st.write(f"**Generated {citation_style} Citation:**", citation)

        with st.expander("Manual Citation"):
            title = st.text_input("Enter paper title:", "My Paper")
            authors = st.text_input("Enter authors (comma-separated):", "Author A, Author B")
            year = st.number_input("Enter year:", 1900, 2025, 2023)
            if st.button("Generate Manual Citation"):
                if citation_style == "APA":
                    citation = f"{authors}. ({year}). {title}."
                elif citation_style == "MLA":
                    authors_str = " and ".join([a.strip() for a in authors.split(",")])
                    citation = f"{authors_str}. \"{title}.\" {year}."
                elif citation_style == "IEEE":
                    authors_str = ", ".join([f"{a.split()[-1]}, {a.split()[0][0]}." for a in authors.split(",")])
                    citation = f"{authors_str}, \"{title},\" {year}."
                st.write(f"**Manual {citation_style} Citation:**", citation)

        st.subheader("Reference Organization")
        if "generated_titles" not in st.session_state:
            st.session_state["generated_titles"] = set()
        if st.button("Suggest Related References"):
            with st.spinner("Fetching references..."):
                initial_suggestions = suggest_references(paper_text)
                st.session_state["generated_titles"] = set(s.split(". ")[1] for s in initial_suggestions if len(s.split(". ")) > 1)
                st.write("**Suggested References (APA):**")
                for sug in initial_suggestions:
                    st.write(f"- {sug}")

        if st.session_state.get("generated_titles"):
            num_more = st.number_input("Generate more references:", min_value=1, max_value=10, value=5)
            if st.button("Generate Additional References"):
                with st.spinner(f"Fetching {num_more} more..."):
                    additional_suggestions = suggest_references(paper_text, offset=len(st.session_state["generated_titles"]) // 2, limit=num_more)
                    new_titles = set(s.split(". ")[1] for s in additional_suggestions if len(s.split(". ")) > 1)
                    st.session_state["generated_titles"] = st.session_state["generated_titles"].union(new_titles)
                    st.write(f"**Additional References (APA):**")
                    for sug in additional_suggestions:
                        st.write(f"- {sug}")

        st.subheader("Cross-Referencing")
        if st.button("Identify Cross-References"):
            with st.spinner("Linking citations..."):
                links = cross_reference_citations(paper_text)
                st.write("**Cross-References:**")
                for link in links:
                    st.write(f"- {link}")

        st.subheader("Citation Network")
        if st.button("Visualize Citation Network"):
            with st.spinner("Building citation network..."):
                doi, _ = extract_metadata_identifiers(paper_text)
                if doi:
                    G = build_citation_network(doi)
                    if G:
                        pos = nx.spring_layout(G)
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        node_x = [pos[node][0] for node in G.nodes()]
                        node_y = [pos[node][1] for node in G.nodes()]
                        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), mode="lines")
                        node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()), 
                                              textposition="top center", marker=dict(size=10))
                        fig = go.Figure(data=[edge_trace, node_trace], 
                                       layout=go.Layout(showlegend=False, title="Citation Network"))
                        st.plotly_chart(fig)
                    else:
                        st.info("No network data available.")
                else:
                    st.warning("No DOI found in paper.")

        st.subheader("Journal Matcher")
        if st.button("Find Suitable Journals"):
            with st.spinner("Matching journals..."):
                prompt = f"Based on this paper excerpt: {paper_text[:2000]}, suggest 3 academic journals suitable for submission. Include journal name and submission URL."
                response = llm_groq.invoke(prompt)
                st.write("**Recommended Journals:**")
                st.markdown(response.content.strip())

    # Data and Code Assistance
    elif options == "🖥️ Data and Code Assistance":
        st.header("🖥️ Data and Code Assistance")
        task = st.radio("Choose a task:", [
            "💻 Generate Code", "📊 Data Analysis", 
            "🔍 Reproducibility Check", "📈 Dataset Analysis", 
            "🌐 Dataset Recommendations", "📓 Export to Jupyter"
        ])
        if task == "💻 Generate Code":
            code_task = st.text_input("Describe the code you need:")
            if code_task:
                with st.spinner("Generating code..."):
                    response = llm_groq.invoke(
                        f"Generate Python code based on this paper: {paper_text[:4000]}\nTask: {code_task}"
                    )
                    st.write("**Generated Code:**", response.content.strip())
        elif task == "📊 Data Analysis":
            if st.button("Analyze Data"):
                with st.spinner("Analyzing data..."):
                    response = llm_groq.invoke(
                        f"Provide data analysis guidance based on this paper: {paper_text[:4000]}"
                    )
                    st.write("**Analysis Guidance:**", response.content.strip())
        elif task == "🔍 Reproducibility Check":
            if st.button("Check Reproducibility"):
                with st.spinner("Checking reproducibility..."):
                    response = llm_groq.invoke(
                        f"Check reproducibility of methods in this paper: {paper_text[:4000]}"
                    )
                    st.write("**Reproducibility Report:**", response.content.strip())
        elif task == "📈 Dataset Analysis":
            st.subheader("Dataset Analysis")
            data_file = st.file_uploader("Upload dataset (CSV/Excel):", type=["csv", "xlsx"])
            if data_file:
                if data_file.size > 10 * 1024 * 1024:
                    st.error("File too large. Upload a file under 10MB.")
                else:
                    with st.spinner("Loading dataset..."):
                        try:
                            if data_file.name.endswith(".csv"):
                                df = pd.read_csv(data_file)
                            else:
                                df = pd.read_excel(data_file)
                            st.write("**Dataset Preview:**")
                            st.dataframe(df.head())
                            st.write("**Basic Statistics:**")
                            st.write(df.describe())
                            if st.button("Visualize Data"):
                                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                                if numeric_cols.size > 0:
                                    fig, ax = plt.subplots()
                                    sns.histplot(df[numeric_cols[0]], kde=True, ax=ax)
                                    plt.title(f"Distribution of {numeric_cols[0]}")
                                    st.pyplot(fig)
                                else:
                                    st.warning("No numeric columns to visualize.")
                        except Exception as e:
                            st.error(f"Failed to load dataset: {e}")
        elif task == "🌐 Dataset Recommendations":
            st.subheader("Dataset Recommendations")
            if st.button("Suggest Datasets"):
                with st.spinner("Finding datasets..."):
                    prompt = f"Based on this paper: {paper_text[:2000]}, suggest 3 open datasets relevant to its topic. Include name, source, and URL."
                    response = llm_groq.invoke(prompt)
                    st.write("**Suggested Datasets:**")
                    st.markdown(response.content.strip())
        elif task == "📓 Export to Jupyter":
            st.subheader("Export to Jupyter")
            code_task = st.text_input("Describe code to export:")
            if code_task and st.button("Generate and Export Notebook"):
                with st.spinner("Creating notebook..."):
                    response = llm_groq.invoke(f"Generate Python code for: {code_task}\nBased on paper: {paper_text[:2000]}")
                    code = response.content.strip()
                    nb = create_notebook(code)
                    nb_data = nbf.writes(nb)
                    st.download_button("Download Notebook", nb_data, "research.ipynb", "application/x-ipynb+json")

    # Visualizations
    elif options == "📊 Visualizations":
        st.header("📊 Visualizations")
        viz_type = st.selectbox("Choose visualization type:", [
            "Histogram (Word Lengths)", "Bar Chart (Sections)", "Key Terms (Abstract)"
        ])
        plt.clf()
        if viz_type == "Histogram (Word Lengths)":
            with st.spinner("Generating histogram..."):
                words = paper_text.split()
                sample_size = min(1000, len(words))
                word_lengths = [len(word.strip(".,!?")) for word in words[:sample_size]]
                plt.hist(word_lengths, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
                plt.xlabel("Word Length")
                plt.ylabel("Frequency")
                plt.title("Distribution of Word Lengths")
                st.pyplot(plt.gcf())
        elif viz_type == "Bar Chart (Sections)":
            with st.spinner("Generating bar chart..."):
                text_lower = paper_text.lower()
                sections = ["introduction", "methods", "results", "discussion"]
                counts = [len(re.findall(rf'\b{section}\b', text_lower)) for section in sections]
                plt.bar(sections, counts, color='lightgreen', edgecolor='black')
                plt.xlabel("Sections")
                plt.ylabel("Occurrences")
                plt.title("Section Mentions in Paper")
                plt.xticks(rotation=45)
                st.pyplot(plt.gcf())
        elif viz_type == "Key Terms (Abstract)":
            with st.spinner("Generating visualization..."):
                words = re.findall(r'\b\w+\b', paper_text.lower())
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
                word_counts = Counter(filtered_words).most_common(10)
                terms, frequencies = zip(*word_counts) if word_counts else (["No terms found"], [1])
                viz_style = st.radio("Visualization style:", ["Bar Chart", "Word Cloud"] if WORDCLOUD_AVAILABLE else ["Bar Chart"], horizontal=True)
                if viz_style == "Bar Chart":
                    plt.bar(terms, frequencies, color='coral', edgecolor='black')
                    plt.xlabel("Key Terms")
                    plt.ylabel("Frequency")
                    plt.title("Top 10 Key Terms in Paper")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                elif viz_style == "Word Cloud" and WORDCLOUD_AVAILABLE:
                    if word_counts:
                        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                            max_words=10, colormap='viridis').generate_from_frequencies(dict(word_counts))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title("Key Terms Word Cloud")
                        st.pyplot(plt.gcf())
                    else:
                        st.write("No significant terms found.")
                if word_counts:
                    abstract = f"This paper focuses on {', '.join(terms[:5])}. It explores key aspects related to {terms[0]} and {terms[1]}, with emphasis on {terms[2]}."
                    st.write("**Paper Abstract (Generated):**", abstract)

    # Paper to Podcast
    elif options == "🎙️ Paper to Podcast":
        st.header("🎙️ Research Paper to Podcast Converter")
        summarize = st.checkbox("Summarize the paper before conversion")
        sentiment_adjustment = st.checkbox("Adjust voice tone based on sentiment")
        min_duration = st.slider("Minimum podcast duration (minutes):", 1, 5, 2)
        if st.button("Generate Podcast"):
            with st.spinner("Processing text..."):
                st.write("Step 1: Extracting text...")
                text_to_convert = paper_text[:2000]
                st.write(f"Text length: {len(text_to_convert)} characters")
                if summarize:
                    with st.spinner("Summarizing content..."):
                        st.write("Step 2: Summarizing text...")
                        text_to_convert = summarize_text(text_to_convert)
                        if text_to_convert is None:
                            st.error("Summarization failed.")
                            st.stop()
                        st.write(f"Summary length: {len(text_to_convert)} characters")
                min_words = min_duration * 120
                st.write(f"Step 3: Checking length (target {min_words} words)...")
                if len(text_to_convert.split()) < min_words:
                    with st.spinner("Expanding content..."):
                        response = llm_groq.invoke(
                            f"Expand this text to at least {min_words} words while keeping its meaning: {text_to_convert}"
                        )
                        text_to_convert = response.content.strip()
                        st.write(f"Expanded length: {len(text_to_convert.split())} words")
                st.write("Step 4: Enhancing text...")
                enhanced_text = enhance_podcast_text(text_to_convert)
                st.write(f"Final text length: {len(enhanced_text.split())} words")
                with st.spinner("Generating audio..."):
                    st.write("Step 5: Converting to audio...")
                    audio_buffer = text_to_speech_gtts(enhanced_text, sentiment_adjustment)
                    if audio_buffer is None:
                        st.error("Audio generation failed.")
                        st.stop()
                    st.success("Podcast generated!")
                    st.audio(audio_buffer, format="audio/mp3")

    # Dashboard
    elif options == "📈 Dashboard":
        st.header("📈 Research Progress Dashboard")
        metrics = {
            "Papers Uploaded": 1 if "paper_text" in st.session_state else 0,
            "Drafts Saved": len(st.session_state.get("drafted_content", "")) // 100,
            "References Collected": len(st.session_state.get("generated_titles", set()))
        }
        fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()), title="Research Progress")
        st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit and Grok")
