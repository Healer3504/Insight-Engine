# 💡 InsightEngine

An AI-powered RAG search engine for documents using Streamlit and Google Gemini's Vertex AI. This application allows users to upload documents (PDF, DOCX, TXT) and ask questions in natural language to get synthesized, context-aware answers.

## ✨ Features

- **Intuitive Frontend:** A clean and modern user interface built with Streamlit.
- **Multi-File Support:** Ingests and processes `.pdf`, `.docx`, and `.txt` files.
- **RAG Pipeline:** Implements a full Retrieval-Augmented Generation pipeline using LangChain.
- **Vector Storage:** Uses ChromaDB to create and store a local vector database from your documents.
- **Advanced AI:** Powered by Google's **Gemini Pro** model via the stable Vertex AI API.

## 🚀 How to Run This Project Locally

### 1. Prerequisites

- Python 3.10+
- A Google Cloud project with the **Vertex AI API** enabled and **Billing** configured.

### 2. Setup

First, clone this repository to your local machine:
```bash
git clone [https://github.com/Healer3504/Insight-Engine.git](https://github.com/Healer3504/Insight-Engine.git)
cd Insight-Engine
