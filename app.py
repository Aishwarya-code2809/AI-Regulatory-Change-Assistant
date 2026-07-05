import streamlit as st
import fitz
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from collections import Counter

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="🏦 AI Regulatory Change Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f6f9fc;
}

.block-container{
    padding-top:2rem;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

.report-box{
    background:white;
    padding:20px;
    border-radius:10px;
    border-left:8px solid #0A5EB0;
}

.success-box{
    background:#e8fff2;
    padding:15px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    llm = Ollama(
        model="llama3.1"
    )

    return embedding_model, llm

embedding_model, llm = load_models()

# ----------------------------------------------------
# LOAD KNOWLEDGE BASE
# ----------------------------------------------------

@st.cache_resource
def load_knowledge_base():

    index = faiss.read_index(
        "vector_db/bank_index.faiss"
    )

    with open("vector_db/chunks.pkl","rb") as f:
        chunks = pickle.load(f)

    with open("vector_db/chunk_metadata.pkl","rb") as f:
        metadata = pickle.load(f)

    return index,chunks,metadata

index,chunks,chunk_metadata = load_knowledge_base()

# ----------------------------------------------------
# PDF FUNCTION
# ----------------------------------------------------

def extract_text_from_pdf(uploaded_file):

    document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    text=""

    for page in document:
        text += page.get_text()

    return text

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/bank-building.png",
        width=80
    )

    st.title("Project Dashboard")

    st.divider()

    st.markdown("### ⚙ Project Information")

    st.write("**Version** : 1.0")

    st.write("**Embedding Model**")

    st.caption("all-MiniLM-L6-v2")

    st.write("**LLM**")

    st.caption("Llama 3.1")

    st.write("**Knowledge Base**")

    st.caption(f"{len(chunks)} Chunks")

    st.write("**Documents**")

    st.caption(f"{len(set(chunk_metadata))} Documents")

    st.success("🟢 System Online")

    st.divider()

    st.info(
        "Upload an RBI Circular and generate an automated Regulatory Impact Assessment Report."
    )

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.title("🏦 AI Regulatory Change Assistant")

st.caption(
    "Enterprise Banking Compliance Automation using Retrieval-Augmented Generation (RAG)"
)

st.divider()

# ----------------------------------------------------
# METRICS
# ----------------------------------------------------

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric(
        "Documents",
        len(set(chunk_metadata))
    )

with col2:
    st.metric(
        "Knowledge Chunks",
        len(chunks)
    )

with col3:
    st.metric(
        "Embedding Model",
        "MiniLM"
    )

with col4:
    st.metric(
        "LLM",
        "Llama3.1"
    )

st.divider()

# ----------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload RBI Circular",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(
        f"Uploaded : {uploaded_file.name}"
    )

    analyze = st.button(
        "🚀 Analyze Circular",
        use_container_width=True
    )
    if analyze:

        progress = st.progress(0)

        status = st.empty()

        # -----------------------------
        # STEP 1
        # -----------------------------

        status.info("📄 Extracting RBI Circular...")

        rbi_text = extract_text_from_pdf(uploaded_file)

        progress.progress(20)

        # -----------------------------
        # STEP 2
        # -----------------------------

        status.info("🧠 Creating Embedding...")

        rbi_embedding = embedding_model.encode([rbi_text])
        rbi_embedding = np.array(rbi_embedding).astype("float32")

        progress.progress(40)

        # -----------------------------
        # STEP 3
        # -----------------------------

        status.info("🔎 Searching Knowledge Base...")

        distances, indices = index.search(
            rbi_embedding,
            k=10
        )

        progress.progress(60)

        # -----------------------------
        # STEP 4
        # -----------------------------

        status.info("📂 Finding Related Documents...")

        matched_documents = []

        context_chunks = []

        similarity_scores = []

        for distance, idx in zip(distances[0], indices[0]):

            doc = chunk_metadata[idx]

            context_chunks.append(chunks[idx])

            similarity = round(
                (1 / (1 + distance)) * 100,
                2
            )

            if doc not in matched_documents:

                matched_documents.append(doc)

                similarity_scores.append(similarity)

        progress.progress(80)

        # -----------------------------
        # DOCUMENTS
        # -----------------------------

        st.divider()

        st.subheader("📂 Matched Internal Documents")

        cols = st.columns(2)

        for i, (doc, score) in enumerate(
            zip(matched_documents, similarity_scores)
        ):

            with cols[i % 2]:

                st.markdown(
                    f"""
<div style="background:white;
padding:18px;
border-radius:12px;
border-left:6px solid #0A5EB0;
margin-bottom:15px;
box-shadow:0 2px 10px rgba(0,0,0,.08);">

<h4>📄 {doc}</h4>

<b>Similarity</b>

{score} %

</div>
""",
                    unsafe_allow_html=True
                )

        progress.progress(90)

        # -----------------------------
        # RBI PREVIEW
        # -----------------------------

        with st.expander("📄 RBI Circular Preview"):

            st.write(rbi_text[:3000])

        # -----------------------------
        # RETRIEVED CHUNKS
        # -----------------------------

        with st.expander("📚 Retrieved Knowledge Base Chunks"):

            for i, idx in enumerate(indices[0]):

                st.markdown(f"### Chunk {i+1}")

                st.write(chunk_metadata[idx])

                st.write(chunks[idx][:600])

                st.divider()

        context = "\n\n".join(context_chunks)

        progress.progress(95)
            # ------------------------------------------
        # GENERATE AI REPORT
        # ------------------------------------------

        status.info("🤖 Generating Regulatory Impact Report...")

        prompt = f"""
You are a Senior Banking Compliance Officer.

Analyze the RBI Circular using ONLY the retrieved internal bank documents.

RBI Circular:
-------------------------------------------------------
{rbi_text[:4000]}
-------------------------------------------------------

Retrieved Internal Documents:
-------------------------------------------------------
{context}
-------------------------------------------------------

Generate a professional Regulatory Impact Assessment Report.

Follow EXACTLY this format.

# REGULATORY IMPACT ASSESSMENT REPORT

## Executive Summary

Briefly summarize the regulatory change.

## Key Regulatory Changes

Mention the major regulatory updates.

## Affected Internal Policies

List affected policies.

## Affected Forms

Mention affected forms.

## Affected SOPs

Mention affected SOPs.

## Operational Impact

Explain how bank operations will change.

## Compliance Risk

State whether risk is:
LOW
MEDIUM
HIGH

Explain why.

## Recommended Actions

Provide bullet points.

## Responsible Departments

Mention departments such as

Compliance

Operations

Risk

IT

Customer Service

## Suggested Timeline

Immediate

30 Days

60 Days

90 Days

Keep the report professional.
"""

        report = llm.invoke(prompt)

        progress.progress(100)

        status.success("✅ Analysis Completed Successfully!")

        st.balloons()

        # ------------------------------------------
        # REPORT
        # ------------------------------------------

        st.divider()

        st.markdown(
            """
# 📋 Regulatory Impact Assessment Report
            """
        )

        # Risk Badge
        report_upper = report.upper()

        if "HIGH" in report_upper:

            st.error("🔴 HIGH RISK")

        elif "MEDIUM" in report_upper:

            st.warning("🟡 MEDIUM RISK")

        else:

            st.success("🟢 LOW RISK")

        st.markdown(report)

        st.divider()

        # ------------------------------------------
        # DOWNLOADS
        # ------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "⬇ Download Report (.txt)",
                report,
                file_name="Regulatory_Impact_Report.txt"
            )

        with col2:

            st.download_button(
                "⬇ Download Report (.md)",
                report,
                file_name="Regulatory_Impact_Report.md"
            )

        st.divider()

        # ------------------------------------------
        # PROJECT STATISTICS
        # ------------------------------------------

        st.subheader("📊 Analysis Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Retrieved Documents",
                len(matched_documents)
            )

        with c2:
            st.metric(
                "Retrieved Chunks",
                len(indices[0])
            )

        with c3:
            st.metric(
                "Knowledge Base",
                len(chunks)
            )

        # ------------------------------------------
        # TECHNICAL DETAILS
        # ------------------------------------------

        with st.expander("⚙ Technical Information"):

            st.write("Embedding Model:", "all-MiniLM-L6-v2")

            st.write("Vector Database:", "FAISS")

            st.write("LLM:", "Llama 3.1")

            st.write("Retrieved Chunks:", len(indices[0]))

            st.write("Knowledge Base Size:", len(chunks))

            st.write("Documents Indexed:", len(set(chunk_metadata)))

        st.divider()

        st.caption(
            "🏦 AI Regulatory Change Assistant | "
            "Powered by Streamlit • FAISS • SentenceTransformers • Llama 3.1"
        )