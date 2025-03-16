# 📚 ResearchPal: AI-Powered Research Assistant

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLaMa--3.3--70B-green)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

ResearchPal is an all-in-one AI-powered research assistant that helps you explore, analyze, and extract insights from academic papers. Upload your research PDFs and unlock a suite of powerful tools designed to boost your productivity and comprehension.

![ResearchPal Demo](https://raw.githubusercontent.com/yourusername/researchpal/main/demo.gif)

## ✨ Features

### 🔍 Research Paper Discovery
- Search for keywords within your papers
- Highlight and extract key information
- Navigate complex research papers effortlessly

### 💬 Interactive Chatbot
- Ask questions directly about your paper
- Get contextual answers powered by LLaMa 3.3
- Maintain conversation history for reference

### ✍️ Writing Assistance
- Generate outlines based on paper structure
- Draft new sections with AI assistance
- Paraphrase text to improve clarity

### 📑 Citation and Reference Management
- Auto-generate citations in APA, MLA, or IEEE formats
- Discover related references and papers
- Identify cross-references within documents

### 🖥️ Data and Code Assistance
- Generate code based on paper methodologies
- Get data analysis guidance
- Verify reproducibility of research methods

### 📊 Visualizations
- View word length distributions
- Analyze section frequency
- Generate word clouds of key terms

### 🎙️ Paper to Podcast
- Convert papers into audio format
- Adjust tone based on content sentiment
- Create engaging podcasts from academic text

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/researchpal.git
cd researchpal

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔑 Environment Setup

1. Create a `.env` file in the project root
2. Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Usage

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser.

## 📋 Requirements

See `requirements.txt` for a full list of dependencies. Main requirements include:

- Python 3.8+
- Streamlit
- Groq
- SentenceTransformers
- PyPDF2
- Transformers
- TextBlob
- gTTS
- WordCloud
- Matplotlib

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework
- [Groq](https://groq.com/) for the LLM API
- [Hugging Face](https://huggingface.co/) for NLP models
- [PyPDF2](https://pythonhosted.org/PyPDF2/) for PDF processing
- All other open-source libraries that made this project possible

---

Made with ❤️ by Aman Tiwari 
