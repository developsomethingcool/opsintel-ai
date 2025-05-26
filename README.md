# opsintel-ai-Document Intelligence Platform

A sophisticated document intelligence platform that provides both naive and agentic RAG (Retrieval-Augmented Generation) capabilities for analyzing PDF documents. Built with Streamlit, LangChain, and LangGraph.

## 🚀 Features

- **Dual RAG Approaches**: Choose between naive RAG for simple queries or agentic RAG for complex question-answering
- **Interactive Chat Interface**: Streamlit-powered web interface with chat bubbles and conversation history
- **Document Ingestion**: Automatic PDF processing and chunking
- **Vector Storage**: ChromaDB-based vector storage with BGE embeddings
- **Smart Retrieval**: Document grading and query rewriting for improved accuracy
- **Multiple LLM Support**: Ollama (default), OpenAI, and Anthropic integrations

## 📁 Project Structure

```
opsintel-ai/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── data/
│   └── raw/                      # Place your PDF files here
├── src/
│   ├── __init__.py
│   ├── embeddings/
│   │   └── embedder.py           # HuggingFace BGE embeddings
│   ├── ingest/
│   │   └── pdf_loader.py         # PDF document loading
│   ├── preprocess/
│   │   └── chunker.py            # Document chunking
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── agentic_rag.py        # Advanced agentic RAG with LangGraph
│   │   └── naive_rag.py          # Simple RAG implementation
│   ├── vectorstore/
│   │   └── chroma_client.py      # ChromaDB vector store management
│   └── db/                       # ChromaDB storage directory
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/developsomethingcool/opsintel-ai
   cd opsintel-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama** (default LLM)
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull qwen2.5:14b  # For agentic RAG
   ```

## 📚 Usage

### Adding Documents

1. Place your PDF files in the `data/raw/` directory
2. The system will automatically process and index them on first run

### Running the Application

```bash
streamlit run app/streamlit_app.py
```

The application will be available at `http://localhost:8501`

### Using Different RAG Approaches

**Agentic RAG (Default in Streamlit app):**
- Intelligent document grading
- Query rewriting for better retrieval
- Multi-step reasoning process
- Handles complex questions effectively

**Naive RAG:**
```python
from src.rag.naive_rag import get_naive_rag

graph = get_naive_rag()
response = graph.invoke({"question": "What does the document say about economics?"})
print(response["answer"])
```

## 🔧 Configuration

### Using OpenAI Instead of Ollama

Replace the LLM initialization in `src/rag/agentic_rag.py`:

```python
from langchain_openai import ChatOpenAI
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Replace ChatOllama instances with:
chat_llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
grader_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

### Using Anthropic Instead of Ollama

Replace the LLM initialization in `src/rag/agentic_rag.py`:

```python
from langchain_anthropic import ChatAnthropic
import os

# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

# Replace ChatOllama instances with:
chat_llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
grader_llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
```

## 🧠 How It Works

### Agentic RAG Workflow

1. **Query Processing**: Initial question analysis
2. **Document Retrieval**: Semantic search in vector store
3. **Document Grading**: Relevance assessment of retrieved documents
4. **Query Rewriting**: Reformulation if documents aren't relevant (up to 3 attempts)
5. **Answer Generation**: Final response based on relevant context

### Vector Store

- **Embeddings**: BGE-Large-EN-v1.5 for high-quality semantic representations
- **Chunking**: Recursive character splitting (1000 chars, 250 overlap)
- **Storage**: ChromaDB with SQLite backend
- **Retrieval**: Top-3 similarity search

## 🚧 Current Limitations & Future Work

### ⚠️ Known Issues

- **German Language Support**: Currently optimized for English documents only
  - BGE embeddings work best with English text
  - Chunking strategies may not be optimal for German grammar
  - LLM prompts are in English

### 🔮 Planned Improvements

1. **Multi-language Support**:
   - Add German-specific embedding models
   - Implement language detection
   - Localized prompts and responses

2. **Enhanced Features**:
   - Document metadata filtering
   - Citation tracking
   - Export conversation history
   - Advanced analytics dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Check the existing issues
- Create a new issue with detailed description
- Include relevant error messages and system information

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [Streamlit](https://streamlit.io/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Embeddings by [BGE](https://huggingface.co/BAAI/bge-large-en-v1.5)
