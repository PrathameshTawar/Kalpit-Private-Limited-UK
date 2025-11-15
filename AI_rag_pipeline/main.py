import os
import streamlit as st
from typing import List

# === LangChain Imports (EXACTLY as per challenge) ===
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama  # Correct import

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# === 1. Create speech.txt if not exists ===
if not os.path.exists("speech.txt"):
    with open("speech.txt", "w", encoding="utf-8") as f:
        f.write(
            "The real remedy is to destroy the belief in the sanctity of the shastras. "
            "How do you expect to succeed if you allow the shastras to continue to be held as sacred and infallible? "
            "You must take a stand against the scriptures. Either you must stop the practice of caste or you must stop believing in the shastras. "
            "You cannot have both. The problem of caste is not a problem of social reform. "
            "It is a problem of overthrowing the authority of the shastras. "
            "So long as people believe in the sanctity of the shastras, they will never be able to get rid of caste. "
            "The work of social reform is like the work of a gardener who is constantly pruning the leaves and branches of a tree without ever attacking the roots. "
            "The real enemy is the belief in the shastras."
        )

# === 2. Load and Split ===
@st.cache_resource
def load_data():
    loader = TextLoader("speech.txt", encoding="utf-8")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    return texts

texts = load_data()

# === 3. Embeddings + Chroma ===
@st.cache_resource
def create_vectorstore(_texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(_texts, embeddings)
    return db

db = create_vectorstore(texts)
retriever = db.as_retriever(search_kwargs={"k": 3})

# === 4. LLM: Ollama (Mistral) ===
llm = Ollama(model="mistral", temperature=0.0)

# === 5. RAG Chain (LCEL - No langchain.chains) ===
template = """Answer the question based **only** on the following context:

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs: List) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# === 6. Streamlit Frontend ===
st.title("AmbedkarGPT - RAG Pipeline")
st.markdown("Ask questions about the speech by Dr. B.R. Ambedkar.")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Generating answer..."):
            try:
                answer = rag_chain.invoke(query.strip())
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
