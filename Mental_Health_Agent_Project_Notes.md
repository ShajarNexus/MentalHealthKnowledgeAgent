# Mental Health Knowledge and Psychoeducation Agent

## Complete Project Notes in Easy English

> **Project purpose:** This application provides general, evidence-grounded mental health education using information stored in its own knowledge base. It retrieves relevant information, generates educational answers using an AI language model, and displays source references.
>
> **Important:** This is an educational assistant. It is not a doctor, therapist, diagnostic system, or personalized treatment service.

---

# 1. What Does the Project Do?

Our project is called **Mental Health Knowledge and Psychoeducation Agent**.

It is an AI-powered application that answers general mental health education questions using a collection of reference documents.

Users can ask questions such as:

- What is anxiety?
- What is depression?
- What is stress?
- How does stress affect mental health?
- Why is sleep important?
- What is emotional wellbeing?
- When should someone seek professional support?

The system searches its knowledge base, retrieves relevant information, checks its relevance, and sends the retrieved information to a language model.

The language model then generates an educational answer.

If the available information is insufficient, the system should avoid generating an unsupported answer.

## Main Objectives

1. Provide general mental health education.
2. Use reference documents as supporting evidence.
3. Retrieve information using RAG.
4. Generate answers using a hosted AI language model.
5. Check whether retrieved information is relevant.
6. Attempt corrective retrieval when the initial search is unsuccessful.
7. Validate generated answers against retrieved evidence.
8. Display source references.
9. Apply safety rules to sensitive questions.

---

# 2. Technologies Used and Why

| Technology | Purpose | Why We Used It |
|---|---|---|
| Python 3.11 | Main programming language | Implements the application's backend logic. |
| Streamlit | Frontend and web application | Provides an interactive interface using Python. |
| LangChain | Connects AI and RAG components | Helps with document loading, splitting, embeddings, vector search, and LLM integration. |
| LangGraph | Agent workflow management | Controls the sequence of agent steps and conditional decisions. |
| Groq API | Hosted AI inference | Allows the application to generate answers without running a language model locally. |
| GPT-OSS 20B | Language model | Generates educational responses using retrieved information. |
| Hugging Face | Embedding model provider | Provides the model used to convert text into numerical vectors. |
| all-MiniLM-L6-v2 | Embedding model | Creates embeddings for questions and document chunks. |
| ChromaDB | Vector database | Stores document chunks and embeddings for semantic retrieval. |
| PyPDF | PDF text extraction | Allows the loader to read PDF documents. |
| TXT files | Knowledge base | Store the mental health reference information. |
| Python-dotenv | Environment variable management | Loads the Groq API key from the local `.env` file. |
| Streamlit Community Cloud | Online hosting | Hosts the application so users can access it through a web browser. |

## Important Difference Between the AI Components

These components have different responsibilities:

- **Hugging Face:** Creates embeddings for searching.
- **ChromaDB:** Stores and retrieves document chunks.
- **Groq:** Provides hosted AI model inference.
- **GPT-OSS 20B:** Generates the written educational answer.
- **LangChain:** Connects the components.
- **LangGraph:** Controls the agent workflow.
- **Streamlit:** Displays the application to users.

---

# 3. What Is RAG?

RAG stands for **Retrieval-Augmented Generation**.

In simple English, RAG means:

1. Retrieve useful information from our own documents.
2. Give the retrieved information to the language model.
3. Generate an answer using that information.

## Why Did We Use RAG?

A language model can generate answers from its training knowledge. However, it may also produce information that is not supported by our project sources.

RAG helps reduce this problem by providing relevant reference information before answer generation.

For example:

A user asks:

"What is anxiety?"

The system:

1. Converts the question into an embedding.
2. Searches ChromaDB.
3. Retrieves relevant chunks from `anxiety.txt` and potentially other documents.
4. Checks retrieval relevance.
5. Sends the retrieved context to Groq.
6. Receives an educational answer.
7. Checks whether the answer is supported by the retrieved context.

**Important:** RAG does not guarantee that an answer is medically correct. It helps ground the response in available sources.

---

# 4. Our Knowledge Base

Our project contains ten mental health reference documents.

Folder structure:

```text
data/
└── mental_health/
    ├── anxiety.txt
    ├── depression.txt
    ├── emotional_wellbeing.txt
    ├── mental_health_awareness.txt
    ├── mental_health_terminology.txt
    ├── professional_support.txt
    ├── psychological_wellbeing.txt
    ├── self_care.txt
    ├── sleep.txt
    └── stress.txt
```

Each document contains educational information about a mental health topic.

The documents can include:

- Topic title
- Source organization
- Reference URL
- Educational content

## Why Did We Create These Documents?

They are the information sources for our RAG system.

Instead of searching the entire internet every time a user asks a question, the system searches our prepared knowledge base.

## Why Are Source Names and URLs Important?

They help identify where information originated.

The application can display source references so users can identify the documents used to support an answer.

A URL stored in a document is a reference. It does not mean the application automatically visits and verifies that website during every question.

---

# 5. Document Loading and Text Chunking

**Main file: `document_loader.py`**

This file prepares documents for the RAG system.

It performs two main tasks.

## A. Loading Documents

The `load_documents()` function:

1. Opens the mental health data folder.
2. Finds supported TXT and PDF files.
3. Reads their contents.
4. Creates LangChain Document objects.
5. Adds metadata such as the source filename.

### Why Do We Need Document Loading?

The system cannot search documents until their content has been read and prepared.

## B. Splitting Documents into Chunks

The `split_documents()` function uses:

`RecursiveCharacterTextSplitter`

Our configuration:

```python
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
```

**Chunk size 700:**

The text is divided into smaller pieces with a target size of approximately 700 characters.

**Chunk overlap 100:**

Neighbouring chunks share approximately 100 characters.

### Why Do We Use Chunking?

Large documents are difficult to search precisely.

Smaller chunks help ChromaDB retrieve the specific sections that relate to a user's question.

### Why Do We Use Overlap?

Overlap helps preserve context when a sentence or important idea is near the end of a chunk.

### Previous Test Result

Our loader test confirmed:

- 10 documents loaded.
- 10 chunks created with the tested documents and current settings.

---

# 6. Embeddings and Hugging Face

**Main file: `rag_pipeline.py`**

An embedding is a numerical representation of text.

For example:

- Question: What is anxiety?
- Document: Anxiety is a feeling of worry or fear.

Although the wording is different, the meanings are related.

Embeddings help the system identify this relationship.

## Where Do We Use Hugging Face?

Our code uses:

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

The configured embedding model is:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

The embedding function creates numerical representations of:

- Document chunks
- User questions

Example configuration:

```python
HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

## Why Did We Choose all-MiniLM-L6-v2?

- It is relatively lightweight.
- It supports semantic search.
- It can run using the CPU.
- It works with LangChain and ChromaDB.

**Remember:** Hugging Face creates embeddings. It does not generate the final chatbot response.

---

# 7. ChromaDB: Our Vector Database

**Main file: `rag_pipeline.py`**

ChromaDB is used to store document chunks and their embeddings.

Our configuration includes:

```python
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "mental_health_knowledge"
```

## How Does ChromaDB Work?

1. The documents are loaded.
2. The documents are divided into chunks.
3. Hugging Face converts the chunks into embeddings.
4. ChromaDB stores the chunks and embeddings.
5. A user submits a question.
6. The question is converted into an embedding.
7. ChromaDB searches for similar document chunks.
8. The retrieved results are returned to the RAG pipeline.

## Why Use Vector Search?

Normal keyword search mainly looks for matching words.

Vector search compares numerical representations of meaning.

For example, a question about feeling worried may retrieve information about anxiety even when the wording is not identical.

---

# 8. Building the Knowledge Base

**Main file: `rag_pipeline.py`**

The main function is:

```python
build_knowledge_base()
```

Its purpose is to prepare the documents for searching.

The general process is:

1. Open or create the ChromaDB collection.
2. Check whether the collection already contains documents.
3. Reuse existing data when appropriate.
4. Load the source documents if the collection needs to be populated.
5. Split the documents into chunks.
6. Create embeddings.
7. Store the chunks in ChromaDB.

The function also supports:

```python
build_knowledge_base(force_rebuild=True)
```

This can be used to intentionally rebuild the knowledge base.

### Why Do We Check Existing Data?

We do not want to add duplicate document chunks every time the application starts.

### Important Deployment Detail

Our `chroma_db/` directory is included in `.gitignore`.

This means the local database is not uploaded to GitHub.

On Streamlit Community Cloud, the application creates its own database from the source documents.

We successfully tested the online application and confirmed:

**Knowledge base ready! 10 chunks indexed.**

This confirms that the online application can load and index the ten source chunks.

---

# 9. Retrieval and Relevance Assessment

**Main file: `rag_pipeline.py`**

## A. Document Retrieval

The function:

```python
retrieve_documents(question, k=TOP_K)
```

searches ChromaDB for document chunks related to the question.

Our configuration:

```python
TOP_K = 5
```

This means the retrieval function requests up to five matching chunks.

## B. Relevance Assessment

The function:

```python
assess_relevance(results)
```

checks the distance values returned by ChromaDB.

Our configured threshold:

```python
MAX_RELEVANCE_DISTANCE = 1.25
```

In our implementation, the best (lowest) returned distance is compared with this threshold.

General interpretation:

- Lower distance: closer vector similarity.
- Higher distance: weaker vector similarity.
- Distance at or below the configured threshold: accepted as relevant by our current rule.

The threshold is a project configuration, not a universal scientific or medical standard.

## Example

User question:

"What is anxiety?"

The system searches ChromaDB and attempts to retrieve relevant chunks.

If the relevance check accepts the results, the agent can continue to answer generation.

If the initial retrieval is not accepted, the agent can attempt corrective retrieval.

**Important limitation:** Our current relevance assessment uses a distance-based heuristic. It does not prove that every returned chunk is useful or that the source itself is factually correct.

---

# 10. Groq API and GPT-OSS 20B

**Main file: `agent.py`**

Our latest project version uses **Groq API** instead of local Ollama.

Previously, we used:

- Ollama
- Qwen 2.5 3B

In the updated version, we use:

- Groq API
- GPT-OSS 20B (`openai/gpt-oss-20b`)

## What Is Groq?

Groq provides hosted AI model inference through an API.

In simple words, our application sends a request to Groq, and Groq runs the selected language model and returns the response.

This means we do not need to run the language model locally using Ollama.

## What Is GPT-OSS 20B?

GPT-OSS 20B is the language model configured in our application.

It generates educational responses using:

- The user's question.
- The retrieved information from our knowledge base.
- Instructions provided through our system prompt.

## Why Did We Change from Ollama to Groq?

Our original local version depended on Ollama running on the computer.

That created a problem for online deployment because the hosted application could not use the Ollama service running on our personal computer.

We changed to Groq so the deployed application could communicate with a hosted language model.

### Current Model Configuration

```python
return ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0.1
)
```

### Important Difference

**Groq provides the model inference service. GPT-OSS 20B is the model.**

Groq is not the embedding model and does not replace ChromaDB.

---

# 11. API Key Management

**Main file: `agent.py`**

Our application needs a Groq API key to communicate with the hosted model.

We use two methods depending on where the application is running.

## A. Local VS Code

The API key is stored in a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=gsk_your_actual_key
```

Python-dotenv loads the key:

```python
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env"
)
```

## B. Streamlit Community Cloud

For online deployment, we add the API key to:

Streamlit Community Cloud → App Settings → Secrets.

Example:

```toml
GROQ_API_KEY = "gsk_your_actual_key"
```

The updated `get_groq_api_key()` function checks:

1. Local environment variables.
2. Streamlit Cloud Secrets if the key is not found locally.

This allows the same application to work in both environments.

## Why Is `.env` Not Uploaded to GitHub?

The `.env` file contains a private API key.

Uploading it publicly could expose the key to other people.

Our `.gitignore` includes:

```gitignore
.env
.env.*
!.env.example
```

The actual API key should remain private.

---

# 12. LangChain and LangGraph

These are two different technologies with different responsibilities.

## A. LangChain

LangChain provides components for building applications that use language models.

In our project, it helps with:

- Document loading.
- Text splitting.
- Embeddings.
- Vector database integration.
- Connecting to Groq.
- Sending prompts to the language model.

Examples from our code:

```python
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_groq import ChatGroq
```

## B. LangGraph

LangGraph controls the agent workflow.

It organizes the application into nodes and edges.

A node performs a particular task.

An edge connects one node to another.

A conditional edge decides which node should execute next.

### Why Did We Use LangGraph?

Our project needs more than a simple question-and-answer sequence.

It must:

- Screen sensitive questions.
- Retrieve information.
- Check relevance.
- Attempt corrective retrieval.
- Stop when evidence is insufficient.
- Generate an answer.
- Validate the generated response.

LangGraph makes these different paths easier to organize.

---

# 13. Complete LangGraph Agent Workflow

**Main file: `agent.py`**

Our updated implementation has eight registered nodes.

| Node | Responsibility |
|---|---|
| `safety_screen` | Checks whether the question contains configured sensitive terms. |
| `retrieve` | Searches the knowledge base for relevant documents. |
| `corrective_retrieval` | Attempts another search using a revised question. |
| `prepare_context` | Combines retrieved document text and source information. |
| `generate` | Uses Groq to generate an educational answer. |
| `reflect` | Checks whether the generated answer is supported by the retrieved context. |
| `insufficient` | Returns an insufficient-evidence response. |
| `human_review` | Returns a safety response for flagged questions. |

## Workflow Explanation

### Step 1: Safety Screening

The first node is:

```python
safety_screen_node()
```

It checks the user's question against a list of configured sensitive terms.

Examples include:

- Suicide
- Self-harm
- Overdose
- Diagnose me
- Which medication?

If a configured term is found, the workflow routes to the human-review response.

Otherwise, it continues to retrieval.

**Important:** This is a basic keyword-based safety screen. It cannot detect every sensitive situation or understand every possible meaning.

### Step 2: Initial Retrieval

The node:

```python
retrieve_node()
```

calls:

```python
retrieve_documents()
```

It retrieves relevant document chunks from ChromaDB and checks their relevance.

### Step 3: Relevance Routing

The function:

```python
relevance_router()
```

decides what happens next.

There are two main possibilities:

- Relevant evidence: Continue to context preparation.
- Insufficient relevance: Attempt corrective retrieval.

### Step 4: Corrective Retrieval

The node:

```python
corrective_retrieval_node()
```

creates a revised query:

```python
revised_question = (
    "Mental health educational information about: "
    + state["question"]
)
```

It searches ChromaDB again with a larger requested number of results.

The system then checks the new results.

If they are relevant, the workflow continues.

If they are still not relevant, it moves to the insufficient-evidence node.

### Step 5: Prepare Context

The node:

```python
prepare_context_node()
```

collects:

- Source filename.
- Retrieval distance.
- Retrieved document content.

It combines the retrieved information into one context string.

It also collects the source names for display.

### Step 6: Generate Answer

The node:

```python
generate_node()
```

sends the following to Groq:

- System instructions.
- User question.
- Retrieved evidence.

Groq uses GPT-OSS 20B to generate the educational answer.

### Step 7: Reflection and Validation

The node:

```python
reflect_node()
```

sends the retrieved context and generated answer to the model for a support check.

The model is instructed to begin with one of two verdicts:

- SUPPORTED
- NOT SUPPORTED

If the verdict is accepted as SUPPORTED, the answer is returned with:

```text
VALIDATED
```

Otherwise, the application returns a message explaining that the generated answer could not be validated.

**Important:** This is an automated model-based check, not independent proof of factual or clinical accuracy.

### Step 8: Insufficient Evidence

The node:

```python
insufficient_node()
```

is used when the retrieval and corrective retrieval attempts do not provide accepted evidence.

The application returns a message explaining that it cannot provide a reliable answer from the available knowledge base.

### Step 9: Human Review

The node:

```python
human_review_node()
```

returns a safety-focused response when the initial keyword screening flags a question.

It does not diagnose or provide personalized clinical instructions.

---

# 14. Overall System Architecture

```text
             MENTAL HEALTH SOURCE DOCUMENTS
                   10 TXT DOCUMENTS
                          |
                          v
                  DOCUMENT LOADER
                 document_loader.py
                          |
                          v
                    TEXT CHUNKING
                 Size: 700 characters
                 Overlap: 100 characters
                          |
                          v
                 HUGGING FACE MODEL
                   all-MiniLM-L6-v2
                          |
                          v
                     CHROMADB
                 Store document vectors
                          |
                          |
                          v
USER QUESTION ------> STREAMLIT APP
                          |
                          v
                    LANGGRAPH AGENT
                       agent.py
                          |
                          v
                   SAFETY SCREENING
                          |
                  +-------+-------+
                  |               |
              Sensitive       Normal query
                  |               |
                  v               v
             HUMAN REVIEW     RETRIEVAL
                                  |
                                  v
                         RELEVANCE CHECK
                                  |
                       +----------+----------+
                       |                     |
                    Relevant            Not relevant
                       |                     |
                       v                     v
                 PREPARE CONTEXT     CORRECTIVE RETRIEVAL
                       |                     |
                       |               +-----+------+
                       |               |            |
                       |            Relevant     Not relevant
                       |               |            |
                       |               |            v
                       |               |       INSUFFICIENT
                       |               |         EVIDENCE
                       |               |
                       +-------+-------+
                               |
                               v
                         GROQ API
                               |
                               v
                         GPT-OSS 20B
                               |
                               v
                         GENERATE ANSWER
                               |
                               v
                         REFLECTION CHECK
                               |
                               v
                      VALIDATION RESULT
                               |
                               v
                  ANSWER + SOURCES + STATUS
                               |
                               v
                        STREAMLIT UI
```

---

# 15. Streamlit Frontend

**Main file: `app.py`**

Streamlit provides the user interface.

It allows users to:

1. Open the application.
2. Enter a mental health question.
3. Submit the question.
4. Wait for the agent to process it.
5. Read the educational answer.
6. View retrieval relevance information.
7. See retrieved document references.
8. Read the reflection and validation status.
9. Inspect retrieved evidence.

## Knowledge Base Button

The application also includes:

**Build / Check Knowledge Base**

This button calls:

```python
build_knowledge_base()
```

It prepares the ChromaDB knowledge base from the source documents.

Our online test successfully displayed:

```text
Knowledge base ready! 10 chunks indexed.
```

## Why Did We Use Streamlit?

- It is easy to build with Python.
- It provides an interactive web interface.
- It does not require a separate React frontend.
- It supports online deployment through Streamlit Community Cloud.

---

# 16. Online Deployment Architecture

Our project now supports both local development and online use.

## Previous Version: Local Ollama

Previously, the application used:

```text
Streamlit
    |
    v
Python Backend
    |
    v
RAG Pipeline
    |
    v
ChromaDB
    |
    v
Ollama
    |
    v
Qwen 2.5 3B
```

The language model ran on the local computer.

The computer needed Ollama installed and running.

## Updated Version: Hosted Groq

Our current architecture is:

```text
             USER
              |
              v
        WEB BROWSER
              |
              v
     STREAMLIT COMMUNITY CLOUD
              |
              v
       STREAMLIT APPLICATION
              |
              v
         PYTHON BACKEND
              |
              v
        LANGGRAPH AGENT
              |
              v
          RAG PIPELINE
              |
              v
           CHROMADB
              |
              v
       RETRIEVED CONTEXT
              |
              v
           GROQ API
              |
              v
         GPT-OSS 20B
              |
              v
       GENERATED ANSWER
              |
              v
        STREAMLIT OUTPUT
```

## Why Did We Change to Groq?

Ollama was running on our personal computer.

When we deployed the application online, the cloud server could not directly access the Ollama service running on our computer.

Groq provides hosted model inference, so our online application can send requests to its API.

### Benefits

- Users do not need Ollama installed.
- The language model does not need to run on the user's computer.
- The application can be accessed through a web browser.
- The same backend workflow can be used locally and online.

### Important Deployment Requirements

- The source documents must be uploaded to GitHub.
- The ChromaDB database must be created in the cloud environment.
- The Groq API key must be configured in Streamlit Cloud Secrets.
- Python dependencies must be included in `requirements.txt`.

The cloud application needs access to the internet to communicate with Groq.

---

# 17. Safety Rules

Mental health is a sensitive subject. Our system is designed for general education only.

The main safety rules are:

1. Do not diagnose a user.
2. Do not prescribe medication.
3. Do not create personalized treatment instructions.
4. Use retrieved information as supporting evidence.
5. Do not invent information when evidence is insufficient.
6. Display source references when available.
7. Flag configured sensitive questions for the human-review response.
8. Encourage qualified professional support when appropriate.

A disclaimer alone is not enough. The system also uses prompts, safety screening, retrieval checks, and reflection.

**Limitation:** Automated screening and model-based validation cannot replace professional human review.

---

# 18. Important Configuration Settings

**Main file: `config.py`**

| Setting | Current Value | Purpose |
|---|---|---|
| `DATA_DIR` | `data/mental_health` | Source document folder |
| `CHROMA_DIR` | `chroma_db` | Local vector database storage |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `COLLECTION_NAME` | `mental_health_knowledge` | ChromaDB collection |
| `CHUNK_SIZE` | `700` | Target text chunk size |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks requested during initial retrieval |
| `MAX_RELEVANCE_DISTANCE` | `1.25` | Relevance distance threshold |

## Language Model Configuration

The current model is configured in `agent.py`:

```python
model="openai/gpt-oss-20b"
```

The API key is retrieved from:

- Local `.env` file.
- Streamlit Community Cloud Secrets.

Ollama and Qwen are not used by the current Groq-based agent.

---

# 19. Important Project Files

| File / Folder | Responsibility |
|---|---|
| `app.py` | Streamlit frontend and application entry point |
| `agent.py` | LangGraph nodes, routing, Groq integration, safety and reflection |
| `rag_pipeline.py` | Embeddings, ChromaDB, retrieval and relevance assessment |
| `document_loader.py` | Loads TXT/PDF files and splits text into chunks |
| `prompts.py` | System and reflection prompts |
| `config.py` | Central project settings |
| `requirements.txt` | Python dependencies |
| `.env` | Local Groq API key; never upload it |
| `.gitignore` | Prevents private and unnecessary files from being committed |
| `data/mental_health/` | Ten mental health reference documents |
| `chroma_db/` | Local vector database storage |
| `tests/test_agent.py` | Agent tests, according to the current test implementation |
| `test_loader.py` | Tests document loading |
| `test_rag.py` | Tests RAG and retrieval |
| `test_groq.py` | Tests Groq API connectivity |
| `Mental_Health_Agent_Project_Notes.md` | Project documentation |

## Why Is the Virtual Environment Not Uploaded?

The virtual environment contains installed Python packages.

It can be recreated using `requirements.txt`.

It is not necessary to upload the entire environment folder to GitHub.

---

# 20. How to Run the Project Locally

The current version uses Groq instead of Ollama.

## Step 1: Open the Project Folder

```powershell
cd I:\AgenticAi\MentalHealthKnowledgeAgent
```

## Step 2: Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

Use `.venv` instead if that is the actual name of your environment folder.

## Step 3: Install Requirements

```powershell
pip install -r requirements.txt
```

## Step 4: Configure the API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_actual_api_key
```

Do not upload this file to GitHub.

## Step 5: Start Streamlit

```powershell
streamlit run app.py
```

The terminal will display the local application address.

Open that address in your browser.

**Ollama is no longer required for the current Groq-based version.**

---

# 21. How to Deploy on Streamlit Community Cloud

Our application is deployed using GitHub and Streamlit Community Cloud.

## Step 1: Upload Project Code to GitHub

Push the updated project files to your GitHub repository.

## Step 2: Connect GitHub to Streamlit Community Cloud

Select the repository and choose:

```text
app.py
```

as the application entry point.

## Step 3: Configure Secrets

Open:

Streamlit Community Cloud → Manage App → Settings → Secrets.

Add:

```toml
GROQ_API_KEY = "gsk_your_actual_api_key"
```

Do not put the real key in your GitHub repository.

## Step 4: Install Dependencies

Streamlit Cloud installs the packages listed in `requirements.txt`.

## Step 5: Build the Knowledge Base

The online application must create its own ChromaDB database from the uploaded source documents.

Our deployment test confirmed:

```text
Knowledge base ready! 10 chunks indexed.
```

## Step 6: Test the Online Chatbot

Try:

```text
What is anxiety?
```

If the API key and retrieval system are working, the application should retrieve relevant information and generate an educational answer.

---

# 22. Tests and Results

The following tests have been performed during development.

## A. Document Loader Test

Reported result:

- 10 source documents loaded.
- 10 chunks created.

This confirmed that document loading and chunking worked with the tested source files and settings.

## B. RAG Test

Reported local test:

- ChromaDB initially contained zero chunks.
- The knowledge base was built.
- Ten chunks were indexed.
- The question `What is anxiety?` retrieved five results.
- The best retrieval distance was approximately `0.523` in a later test.
- The relevance assessment accepted the retrieved results.

## C. Groq API Test

The Groq connection was tested successfully.

Configured model:

```text
openai/gpt-oss-20b
```

The model returned an educational explanation of anxiety.

## D. Online Deployment Test

The Streamlit Community Cloud application was tested.

Reported results:

- Application opened online.
- Knowledge base successfully indexed ten chunks.
- Groq API key configuration was fixed using Streamlit Cloud Secrets.
- The chatbot successfully generated an answer online.

These results confirm that the main online question-answering workflow is functioning.

Further testing is still needed for all safety, insufficient-evidence, multi-document retrieval, and validation scenarios.

---

# 23. Test Scenarios for Demonstration

| Test Type | Example Question | Expected Behaviour |
|---|---|---|
| Relevant question | What is anxiety? | Retrieve relevant anxiety information and generate an answer. |
| Multiple-document question | Explain the relationship between stress, anxiety and depression. | Retrieve relevant information from multiple documents where available. |
| Sleep-related question | Why is sleep important for mental health? | Retrieve sleep-related information. |
| Emotional wellbeing | How can stress affect emotional wellbeing? | Retrieve related information and generate an educational explanation. |
| Insufficient evidence | Explain how to repair a car engine. | Return an insufficient-evidence response. |
| Out-of-scope question | What is the current price of a phone? | Avoid presenting unrelated mental health information as an answer. |
| Safety-sensitive question | Diagnose my depression and tell me which medicine to take. | Trigger the configured human-review response. |

Always record the actual result of each test. Do not claim that a test passed until it has been performed.

---

# 24. Common Viva Questions and Answers

### Q1. What is the main purpose of your project?

Our project is a mental health education agent. It searches a knowledge base of reference documents and provides general educational answers with source references.

### Q2. What is RAG?

RAG means Retrieval-Augmented Generation. It retrieves information from documents first and then uses that information to generate an answer.

### Q3. Why did you use RAG?

We used RAG to ground the language model's responses in our own mental health reference documents.

### Q4. What is an embedding?

An embedding is a numerical representation of text that helps compare the meaning of questions and documents.

### Q5. Why did you use Hugging Face?

We used Hugging Face's all-MiniLM-L6-v2 model to generate embeddings for semantic search.

### Q6. Why did you use ChromaDB?

ChromaDB stores document chunks and embeddings and retrieves information similar to a user's question.

### Q7. Why did you use LangChain?

LangChain provides components for document loading, text splitting, embeddings, vector storage and language model integration.

### Q8. Why did you use LangGraph?

LangGraph manages our agent workflow using nodes, edges and conditional routing.

### Q9. What is a node?

A node is an individual step in the agent workflow. For example, retrieval, answer generation and reflection are nodes.

### Q10. What is an edge?

An edge connects one node to another and determines the next step in the workflow.

### Q11. What is corrective retrieval?

Corrective retrieval is another retrieval attempt using a revised question when the initial search does not return accepted evidence.

### Q12. What is the role of Groq?

Groq provides hosted AI model inference. Our application sends requests to its API and receives model responses.

### Q13. Which language model are you using?

We currently use GPT-OSS 20B through the Groq API.

### Q14. Why did you replace Ollama?

Ollama was running on our local computer. We changed to Groq because we wanted to deploy our application online without requiring a local Ollama service.

### Q15. Is Groq the embedding model?

No. Groq is the hosted inference provider. We use all-MiniLM-L6-v2 through Hugging Face for embeddings.

### Q16. Does ChromaDB generate answers?

No. ChromaDB retrieves document chunks. The language model generates the written response.

### Q17. What is TOP_K = 5?

It means the initial retrieval function requests up to five matching document chunks.

### Q18. What is the relevance threshold?

It is a configured distance limit used by our retrieval pipeline to decide whether the retrieved results are sufficiently similar to the question.

### Q19. What happens when the system cannot find relevant information?

It attempts corrective retrieval. If that also fails the relevance check, it returns an insufficient-evidence response.

### Q20. What is reflection in your project?

Reflection is an additional model-based check that evaluates whether the generated answer is supported by the retrieved context.

### Q21. Does reflection guarantee correctness?

No. It is an automated validation step. It cannot guarantee that every statement is factually or medically correct.

### Q22. Why did you use Streamlit?

Streamlit provides a simple Python-based interface where users can enter questions and view answers, sources and validation information.

### Q23. Can the project work without Ollama?

Yes. The current version uses Groq API, so Ollama is no longer required.

### Q24. How does the online application access the Groq API key?

The key is stored securely in Streamlit Community Cloud Secrets. Locally, it is stored in the `.env` file.

### Q25. Can your project diagnose depression?

No. It is an educational assistant, not a diagnostic system or a replacement for professional mental health care.

---

# 25. Short Explanation to Tell Your Teacher

"Our project is a Mental Health Knowledge and Psychoeducation Agent. We prepared ten mental health reference documents and created a RAG pipeline to search those documents.

First, our document loader reads TXT and PDF files and divides their content into smaller chunks. We use Hugging Face's all-MiniLM-L6-v2 model to convert the chunks into embeddings. ChromaDB stores these embeddings and retrieves relevant information when a user asks a question.

We use LangChain to connect the RAG components and LangGraph to control the agent workflow. Our agent performs safety screening, retrieves documents, checks relevance, attempts corrective retrieval when needed, and prepares the retrieved context.

For answer generation, we use GPT-OSS 20B through the Groq API instead of running a model locally with Ollama. We also use a reflection step to check whether the generated answer is supported by the retrieved information.

Streamlit provides the frontend, and Streamlit Community Cloud hosts our application online.

The project is designed for general mental health education only. It does not diagnose conditions or prescribe treatments."

---

# 26. Important Points to Remember

| Component | Easy Meaning |
|---|---|
| Python | Main programming language |
| Streamlit | Frontend and web application |
| RAG | Retrieve information first, then generate an answer |
| LangChain | Connects RAG and AI components |
| LangGraph | Controls agent workflow and decisions |
| Hugging Face | Provides the embedding model |
| all-MiniLM-L6-v2 | Converts text into embeddings |
| ChromaDB | Stores and retrieves document chunks |
| Groq | Hosted AI inference service |
| GPT-OSS 20B | Generates educational answers |
| Chunking | Divides documents into smaller pieces |
| Corrective retrieval | Attempts another search |
| Reflection | Checks whether an answer is supported by retrieved context |
| Human review | Safety response for flagged questions |
| `.env` | Stores the local API key |
| Streamlit Secrets | Stores the online API key |
| GitHub | Stores the project source code |
| Streamlit Community Cloud | Hosts the application online |

---

# 27. Final Project Summary

Our Mental Health Knowledge and Psychoeducation Agent combines RAG, LangChain, LangGraph, Hugging Face embeddings, ChromaDB and a hosted Groq language model.

It uses ten mental health reference documents to retrieve supporting information and generate educational answers.

The agent includes relevance checking, corrective retrieval, reflection and safety screening.

The application is built with Streamlit and deployed online using Streamlit Community Cloud.

**The most important idea to remember:**

Our system does not simply ask an AI model to answer a question from its own knowledge. It first searches our mental health knowledge base, retrieves supporting information, and then uses that information to generate an educational response.

**Final safety reminder:** The system is designed for education, not diagnosis, personalized treatment or emergency mental health care.