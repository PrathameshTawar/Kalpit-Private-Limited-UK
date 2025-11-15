# AmbedkarGPT - RAG Pipeline

This project implements a Retrieval Augmented Generation (RAG) pipeline using LangChain to answer questions based on a provided speech text. It leverages HuggingFace embeddings, ChromaDB for vector storage, and the Ollama (Mistral) Large Language Model for generating answers.

## Features

- **Text Loading and Splitting**: Loads text from `speech.txt` and splits it into manageable chunks.
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for generating text embeddings.
- **Vector Store**: Utilizes ChromaDB to store and retrieve document embeddings.
- **Ollama Integration**: Connects to an Ollama instance running the Mistral model for language generation.
- **Streamlit Web Interface**: Provides a user-friendly web interface to ask questions about the speech content.

## Setup

### Prerequisites

- Python 3.8+
- Docker (or a local Ollama installation)

### 1. Clone the repository

```bash
git clone https://github.com/abhishekkamble12/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Ollama with Mistral

Ensure you have Ollama installed and running. If not, you can download it from [ollama.ai](https://ollama.ai/).

Pull the Mistral model:

```bash
ollama pull mistral
```

Ensure the Ollama server is running in the background.

### 4. Prepare `speech.txt`

The `main.py` script will automatically create a `speech.txt` file with a default text if it doesn't exist. You can replace the content of `speech.txt` with any text you wish to use for the RAG pipeline.

### 5. Run the Application

```bash
streamlit run main.py
```

The application will start a web server, and you can access the interface in your browser.

## Usage

Open the provided URL in your browser. You will see a simple interface with a text input for questions and an "Ask" button.

Type your question and click "Ask". The RAG pipeline will retrieve relevant context from `speech.txt` and generate an answer using the Mistral LLM.

## Example Interaction

Enter: "What is the real remedy to destroy the belief in the sanctity of the shastras?"

Output: The real remedy is to destroy the belief in the sanctity of the shastras. You must take a stand against the scriptures. Either you must stop the practice of caste or you must stop believing in the shastras.
