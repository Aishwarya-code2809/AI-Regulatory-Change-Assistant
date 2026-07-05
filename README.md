# 🏦 AI Regulatory Change Assistant

An AI-powered Regulatory Change Assistant that automates the analysis of RBI circulars and identifies their impact on internal banking documents using Retrieval-Augmented Generation (RAG).

---

## 📌 Overview

Banks frequently receive regulatory updates from the Reserve Bank of India (RBI). Compliance teams must manually review these circulars and determine which internal policies, SOPs, controls, and customer forms require updates.

This project automates that process by combining semantic search with a Large Language Model (Llama 3.1) to generate a comprehensive Regulatory Impact Assessment Report.

---

## 🚀 Features

- 📄 Upload RBI Circular (PDF)
- 📑 Automatic PDF Text Extraction
- ✂️ Intelligent Text Chunking
- 🧠 Semantic Embeddings using Sentence Transformers
- 🔍 FAISS Vector Similarity Search
- 🤖 Llama 3.1 Powered Report Generation
- 📂 Identification of Impacted Policies, SOPs, Forms and Controls
- 📋 Regulatory Impact Assessment Report
- 🌐 Interactive Streamlit Web Application

---

## 🏗️ System Architecture

```
                 RBI Circular (PDF)
                         │
                         ▼
                PDF Text Extraction
                         │
                         ▼
                   Text Chunking
                         │
                         ▼
          Sentence Transformer Embeddings
                         │
                         ▼
                FAISS Vector Database
                         │
                         ▼
          Retrieve Relevant Documents
                         │
                         ▼
        Llama 3.1 (via Ollama)
                         │
                         ▼
      Regulatory Impact Assessment Report
                         │
                         ▼
              Streamlit Dashboard
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- Sentence Transformers
- FAISS
- Ollama
- Llama 3.1
- PyMuPDF (fitz)
- NumPy

---

## 📂 Project Structure

```
AI-Regulatory-Change-Assistant

│
├── data
│   ├── Policies
│   ├── Controls
│   ├── Forms
│   ├── Regulations
│   └── SOPS
│
├── notebooks
│   ├── 01_pdf_loader.ipynb
│   ├── 02_chunking.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_faiss.ipynb
│   ├── 05_llama3.ipynb
│   ├── 06_build_knowledge_base.ipynb
│   └── 07_regulatory_assistant.ipynb
│
├── vector_db
│   ├── bank_index.faiss
│   ├── chunks.pkl
│   └── chunk_metadata.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Regulatory-Change-Assistant.git
```

Move into the project

```bash
cd AI-Regulatory-Change-Assistant
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download Llama 3.1

```bash
ollama pull llama3.1
```

Run the application

```bash
streamlit run app.py
```

---

## 📋 Workflow

1. Upload an RBI Circular
2. Extract text from PDF
3. Generate semantic embeddings
4. Search FAISS knowledge base
5. Retrieve relevant internal documents
6. Pass retrieved context to Llama 3.1
7. Generate Regulatory Impact Assessment Report

---

## 🎯 Sample Output

The application automatically generates:

- Executive Summary
- Key Regulatory Changes
- Affected Internal Policies
- Affected SOPs
- Affected Forms
- Compliance Risk
- Recommended Actions
- Responsible Departments
- Suggested Implementation Timeline

---

## 💡 Future Improvements

- PDF & DOCX report export
- Multi-user authentication
- Department-wise dashboards
- Integration with RBI notification feeds
- Cloud deployment
- Support for additional regulatory authorities

---

## 👩‍💻 Author

**Aishwarya B S**

Computer Science Engineering Student

Dayananda Sagar University

---

## ⭐ If you found this project useful, consider giving it a star!
