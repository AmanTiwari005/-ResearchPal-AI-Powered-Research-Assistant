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
from wordcloud import WordCloud
from textblob import TextBlob
from gtts import gTTS
from io import BytesIO

# Initialize GROQ chat model
def init_groq_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please set it in Streamlit Cloud secrets.")
        raise ValueError("GROQ_API_KEY not found")
    try:
        # Explicitly initialize Groq client first to test
        groq_client = Groq(api_key=groq_api_key)
        st.write("Groq client initialized successfully")
        # Then pass to ChatGroq
        return ChatGroq(
            api_key=groq_api_key,  # Changed to 'api_key' to match groq.Client expectation
            model="llama-3.3-70b-versatile",  # Changed 'model_name' to 'model'
            temperature=0.2
        )
    except Exception as e:
        st.error(f"Failed to initialize Groq model: {str(e)}")
        raise

# Load Grok model
if "llm_groq" not in st.session_state:
    try:
        st.session_state["llm_groq"] = init_groq_model()
    except ValueError as e:
        st.stop()

llm_groq = st.session_state["llm_groq"]

# Check if GPU is available (will be CPU on Streamlit Cloud)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"**Using device:** {device}")

# Load models with GPU support (will default to CPU on Streamlit Cloud)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Function to extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to split text into paragraphs
def split_into_paragraphs(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    paragraphs = []
    current_paragraph = ""
    for line in lines:
        if line.endswith(".") or line.endswith(":") or line.endswith("!"):
            current_paragraph += " " + line
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
        else:
            current_paragraph += " " + line
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    return paragraphs

# Function to search for keywords in the paper
def search_keywords(paper_text, query):
    paragraphs = split_into_paragraphs(paper_text)
    matches = []
    for i, paragraph in enumerate(paragraphs):
        if re.search(re.escape(query), paragraph, re.IGNORECASE):
            matches.append((i, paragraph))
    return matches

# Function to highlight keywords in the text
def highlight_keywords(text, query):
    highlighted_text = re.sub(
        f"({re.escape(query)})", 
        r"**\1**", 
        text, 
        flags=re.IGNORECASE
    )
    return highlighted_text

# Function to extract DOI or title
def extract_metadata_identifiers(paper_text):
    doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
    doi_match = re.search(doi_pattern, paper_text)
    doi = doi_match.group(0) if doi_match else None
    lines = paper_text.split("\n")
    title = next((line.strip() for line in lines if len(line.strip()) > 10 and line.strip()[0].isupper()), "Untitled")
    return doi, title

# Function to fetch metadata and format citation from multiple APIs
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

# Function: Suggest related references
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
        with st.spinner("Generating additional suggestions with AI..."):
            prompt = (
                f"Based on this paper excerpt: {paper_text[:2000]}, suggest {limit - len(suggestions)} "
                f"related references in APA style. Return each as a single line."
            )
            response = llm_groq.invoke(prompt)
            ai_suggestions = response.content.strip().split("\n")
            suggestions.extend(ai_suggestions[:limit - len(suggestions)])

    return suggestions[:limit]

# Function: Placeholder for cross-referencing
def cross_reference_citations(paper_text):
    citation_pattern = r"\[\d+\]|\(\w+,\s*\d{4}\)"
    matches = re.findall(citation_pattern, paper_text)
    if matches:
        return [f"Found citation: {match}" for match in matches[:5]]
    return ["No clear cross-references identified."]

# Podcast Converter Functions
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
        
        tts = gTTS(text=text, lang='en', slow=(speech_rate == 'slow'))
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
    closing = "That’s all for today’s episode. Thank you so much for joining me on this journey through the research. Stay curious, and until next time, goodbye!"
    
    words = text.split()
    mid_point = len(words) // 2
    first_half = " ".join(words[:mid_point])
    second_half = " ".join(words[mid_point:])
    
    return f"{greeting}
 {first_half} {middle} {second_half} {closing}"

# Streamlit App
st.title("📚 Research Paper Management and Chatbot")
st.markdown("Welcome to your AI-powered research assistant! Explore, summarize, and interact with your personal research paper effortlessly.")

# Sidebar for navigation and file upload
st.sidebar.title("⚙️ Navigation")
uploaded_file = st.sidebar.file_uploader("📤 Upload your research paper (PDF):", type="pdf")
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        paper_text = extract_text_from_pdf(uploaded_file)
        st.session_state["paper_text"] = paper_text
    st.sidebar.success("✅ File uploaded successfully!")
else:
    st.sidebar.warning("⚠️ Please upload a research paper to proceed.")

options = st.sidebar.radio("Choose a feature:", [
    "🔍 Research Paper Discovery", "💬 Interactive Chatbot", "✍️ Writing Assistance",
    "📑 Citation and Reference Management", "🖥️ Data and Code Assistance", 
    "📊 Visualizations", "🎙️ Paper to Podcast"
])

if "paper_text" not in st.session_state:
    st.warning("⚠️ Please upload a research paper first.")
else:
    paper_text = st.session_state["paper_text"]

    # 1. Research Paper Discovery
    if options == "🔍 Research Paper Discovery":
        st.header("🔍 Research Paper Discovery")
        query = st.text_input("Enter keywords to search within your paper:")
        if query:
            with st.spinner("Searching within your paper..."):
                matches = search_keywords(paper_text, query)
                if matches:
                    st.write(f"**Found {len(matches)} matches for '{query}':**")
                    for i, (para_num, paragraph) in enumerate(matches):
                        st.write(f"**Match {i + 1} (Paragraph {para_num + 1}):**")
                        st.write(highlight_keywords(paragraph, query))
                else:
                    st.warning(f"No matches found for '{query}'.")

    # 2. Interactive Chatbot
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

    # 3. Writing Assistance
    elif options == "✍️ Writing Assistance":
        st.header("✍️ Writing Assistance")
        task = st.radio("Choose a task:", ["📝 Generate Outline", "✍️ Draft and Paraphrase"])
        
        if task == "📝 Generate Outline":
            use_uploaded_paper = st.toggle("Base outline on uploaded paper?", value=True)
            if use_uploaded_paper:
                if "paper_text" not in st.session_state:
                    st.warning("⚠️ Please upload a research paper first to use this option.")
                else:
                    topic = st.text_input("Enter a specific focus (optional, refines outline from paper):")
                    if st.button("Generate Outline from Paper"):
                        with st.spinner("Generating outline from uploaded paper..."):
                            prompt = f"Generate a research paper outline based on this content: {paper_text[:4000]}"
                            if topic:
                                prompt += f" with a focus on: {topic}"
                            response = llm_groq.invoke(prompt)
                            st.write("**Outline (Based on Uploaded Paper):**", response.content.strip())
            else:
                topic = st.text_input("Enter topic for new research paper:")
                if topic:
                    if st.button("Generate Outline for New Paper"):
                        with st.spinner("Generating outline for new paper..."):
                            response = llm_groq.invoke(
                                f"Generate a detailed research paper outline for a new paper on the topic: {topic}"
                            )
                            st.write("**Outline (New Paper):**", response.content.strip())
                else:
                    st.info("Please enter a topic to generate an outline for a new paper.")
                    
        elif task == "✍️ Draft and Paraphrase":
            st.subheader("✍️ Draft and Paraphrase")
            if "paper_text" not in st.session_state:
                st.warning("⚠️ Please upload a research paper first to proceed.")
            else:
                if "drafted_content" not in st.session_state:
                    st.session_state["drafted_content"] = ""
                st.write("### Draft Section")
                section = st.text_input("Enter the section (e.g., Introduction, Methodology):", key="draft_section_input")
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
                if st.session_state["drafted_content"]:
                    text_to_paraphrase = st.text_area(
                        "Text to paraphrase (auto-filled from draft):",
                        value=st.session_state["drafted_content"],
                        key="paraphrase_input_auto"
                    )
                else:
                    text_to_paraphrase = st.text_area(
                        "Enter text to paraphrase (or generate a draft first):",
                        key="paraphrase_input_manual"
                    )
                if text_to_paraphrase and st.button("Paraphrase Text"):
                    with st.spinner("Paraphrasing text..."):
                        response = llm_groq.invoke(
                            f"Paraphrase the following text: {text_to_paraphrase}"
                        )
                        paraphrased_text = response.content.strip()
                        st.write("**Paraphrased Text:**")
                        st.write(paraphrased_text)

    # 4. Citation and Reference Management
    elif options == "📑 Citation and Reference Management":
        st.header("📑 Citation and Reference Management")
        st.subheader("Auto-Citation")
        citation_style = st.selectbox("Select Citation Style:", ["APA", "MLA", "IEEE"])
        if st.button("Generate Citation for This Paper"):
            with st.spinner("Fetching citation from multiple APIs..."):
                citation = get_citation_from_apis(paper_text, style=citation_style)
                st.write(f"**Generated {citation_style} Citation:**", citation)

        with st.expander("Manual Citation"):
            st.write("Or manually enter details:")
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
            with st.spinner("Fetching initial references from multiple APIs..."):
                initial_suggestions = suggest_references(paper_text, offset=0)
                st.session_state["generated_titles"] = set(s.split(". ")[1] for s in initial_suggestions if len(s.split(". ")) > 1)
                st.write("**Suggested References (APA):**")
                for sug in initial_suggestions:
                    st.write(f"- {sug}")

        if st.session_state.get("generated_titles"):
            num_more = st.number_input("Generate more references (number):", min_value=1, max_value=10, value=5, step=1)
            if st.button("Generate Additional References"):
                with st.spinner(f"Fetching {num_more} more references..."):
                    additional_suggestions = suggest_references(paper_text, offset=len(st.session_state["generated_titles"]) // 2, limit=num_more)
                    new_titles = set(s.split(". ")[1] for s in additional_suggestions if len(s.split(". ")) > 1)
                    st.session_state["generated_titles"] = st.session_state["generated_titles"].union(new_titles)
                    st.write(f"**Additional Suggested References (APA):**")
                    for sug in additional_suggestions:
                        st.write(f"- {sug}")

        st.subheader("Cross-Referencing")
        if st.button("Identify Cross-References"):
            with st.spinner("Linking related citations..."):
                links = cross_reference_citations(paper_text)
                st.write("**Cross-References:**")
                for link in links:
                    st.write(f"- {link}")

    # 5. Data and Code Assistance
    elif options == "🖥️ Data and Code Assistance":
        st.header("🖥️ Data and Code Assistance")
        task = st.radio("Choose a task:", ["💻 Generate Code", "📊 Data Analysis", "🔍 Reproducibility Check"])
        if task == "💻 Generate Code":
            code_task = st.text_input("Describe the code you need based on your paper:")
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

    # 6. Visualizations
    elif options == "📊 Visualizations":
        st.header("📊 Visualizations")
        viz_type = st.selectbox("Choose visualization type:", ["Histogram (Word Lengths)", "Bar Chart (Sections)", "Key Terms (Abstract)"])
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
            with st.spinner("Generating abstract visualization..."):
                words = re.findall(r'\b\w+\b', paper_text.lower())
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
                word_counts = Counter(filtered_words).most_common(10)
                terms, frequencies = zip(*word_counts) if word_counts else (["No terms found"], [1])
                viz_style = st.radio("Visualization style:", ["Bar Chart", "Word Cloud"], horizontal=True)
                if viz_style == "Bar Chart":
                    plt.bar(terms, frequencies, color='coral', edgecolor='black')
                    plt.xlabel("Key Terms")
                    plt.ylabel("Frequency")
                    plt.title("Top 10 Key Terms in Paper")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                else:
                    if word_counts:
                        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                            max_words=10, colormap='viridis').generate_from_frequencies(dict(word_counts))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title("Key Terms Word Cloud")
                        st.pyplot(plt.gcf())
                    else:
                        st.write("No significant terms found to generate a word cloud.")
                if word_counts:
                    abstract = f"This paper focuses on {', '.join(terms[:5])}. It explores key aspects related to {terms[0]} and {terms[1]}, with emphasis on {terms[2]}."
                    st.write("**Paper Abstract (Generated):**", abstract)

    # 7. Paper to Podcast
    elif options == "🎙️ Paper to Podcast":
        st.header("🎙️ Research Paper to Podcast Converter")
        st.write("Convert your research paper into an audio podcast!")
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

                st.write("Step 4: Enhancing text with greetings...")
                enhanced_text = enhance_podcast_text(text_to_convert)
                st.write(f"Final text length: {len(enhanced_text.split())} words")

                with st.spinner("Generating podcast audio..."):
                    st.write("Step 5: Converting to audio...")
                    audio_buffer = text_to_speech_gtts(enhanced_text, sentiment_adjustment=sentiment_adjustment)
                    if audio_buffer is None:
                        st.error("Audio generation failed.")
                        st.stop()
                    st.success("Podcast generated successfully!")
                    st.audio(audio_buffer, format="audio/mp3")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit and Grok")
