# 🧠 Mental Health Knowledge and Psychoeducation Agent

An evidence-grounded AI agent that provides general mental health education using **Retrieval-Augmented Generation (RAG)** and **LangGraph**.

The system retrieves information from a collection of mental health documents, checks whether the retrieved information is relevant, generates an educational answer using a hosted LLM through Groq, and validates whether the answer is supported by the retrieved evidence.

## 📌 Project Overview

The Mental Health Knowledge and Psychoeducation Agent is designed to help users understand common mental health topics in simple language.

Users can ask questions about topics such as:

- Anxiety
- Depression
- Stress management
- Sleep and mental wellbeing
- Emotional wellbeing
- Mental health terminology
- Professional mental health support

Instead of generating answers only from the language model's general knowledge, the system first retrieves relevant information from its knowledge base.

## 🎯 Project Objectives

- Build a mental health knowledge base using credible educational sources.
- Apply RAG to retrieve relevant information before generating answers.
- Use LangGraph to manage the agent's workflow.
- Check the relevance of retrieved documents.
- Perform corrective retrieval when the initial results are not relevant.
- Generate answers grounded in retrieved evidence.
- Reflect on and validate generated answers against the retrieved context.
- Display source references and retrieved evidence.
- Flag sensitive questions for human review.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| LangChain | RAG components and LLM integration |
| LangGraph | Agent workflow and decision-making |
| Hugging Face | Text embedding model |
| all-MiniLM-L6-v2 | Converts text into numerical vectors |
| ChromaDB | Stores document embeddings and supports similarity search |
| Groq API | Provides hosted LLM access |
| GPT-OSS 20B | Generates educational answers through Groq |
| python-dotenv | Loads local environment variables |

## ⚙️ How the System Works

The project uses a Retrieval-Augmented Generation (RAG) workflow.

### 1. Document Loading

The system reads mental health educational documents from:

`data/mental_health/`

It supports TXT and PDF documents.

### 2. Text Chunking

Large documents are divided into smaller text sections called chunks.

This helps the system retrieve specific information instead of searching through entire documents.

### 3. Text Embeddings

The Hugging Face model `all-MiniLM-L6-v2` converts text chunks into numerical vectors.

These vectors represent the meaning of the text.

### 4. Vector Database

ChromaDB stores the document vectors and their associated text.

When a user asks a question, the system searches for semantically similar document chunks.

### 5. Relevance Checking

The system checks whether the retrieved documents are relevant to the user's question.

If the retrieved information is not relevant, the agent can attempt corrective retrieval.

If sufficient evidence is still unavailable, it returns an insufficient-evidence response.

### 6. Answer Generation

The retrieved information is passed to the hosted language model:

**Provider:** Groq API  
**Model:** `openai/gpt-oss-20b`

The model generates an educational answer using the retrieved context.

### 7. Reflection and Validation

The agent checks whether the generated answer is supported by the retrieved information.

It reports a validation status:

- `VALIDATED` – The reflection step judged the answer to be supported by the retrieved context.
- `NOT VALIDATED` – The answer did not pass that support check.

This is an automated check, not a guarantee that every statement is medically or factually correct.

### 8. Human Review

Sensitive questions can be flagged for human review.

The application is an educational system, not a replacement for a qualified mental health professional.

## 🔄 Project Workflow

```text
Mental Health Documents
          |
          v
   Document Loading
          |
          v
      Text Chunking
          |
          v
   Hugging Face Embeddings
          |
          v
       ChromaDB
          |
          v
      User Question
          |
          v
     LangGraph Agent
          |
          v
    Safety Screening
          |
          v
   Retrieve Relevant Data
          |
          v
    Relevance Checking
          |
          v
  Corrective Retrieval
          |
          v
   Context Preparation
          |
          v
  Groq API - GPT-OSS 20B
          |
          v
    Answer Generation
          |
          v
 Reflection and Validation
          |
          v
     Final Response
          |
          v
  Streamlit User Interface
```

## 📂 Project Structure

```text
MentalHealthKnowledgeAgent/
│
├── app.py
├── agent.py
├── rag_pipeline.py
├── document_loader.py
├── prompts.py
├── config.py
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   └── mental_health/
│       ├── anxiety.txt
│       ├── depression.txt
│       ├── stress.txt
│       ├── emotional_wellbeing.txt
│       └── other mental health documents...
│
├── tests/
│   └── test_agent.py
│
├── logs/
├── chroma_db/
│
└── README.md
```

**Important:** `.env`, `chroma_db/`, virtual environments, and Python cache files should not be uploaded to GitHub. The mental health documents inside `data/mental_health/` should be included in the repository.

## 💻 Installation and Local Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ShajarNexus/MentalHealthKnowledgeAgent.git
```

### Step 2: Open the Project Folder

```bash
cd MentalHealthKnowledgeAgent
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 4: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 5: Configure the Groq API Key

Create a file named `.env` in the main project folder, alongside `app.py`.

Add:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Get an API key from [Groq Console](https://console.groq.com/).

**Important:** Never upload your real API key to GitHub or share it publicly.

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Online Deployment

The application can be deployed using **Streamlit Community Cloud**.

Deployment steps:

1. Upload the project code to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select `app.py` as the application entry point.
5. Open the application settings and select **Secrets**.
6. Add your Groq API key:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

7. Save the secrets and deploy the application.
8. Open the deployed application.
9. Click **Build / Check Knowledge Base**.
10. Ask a mental health education question.

Streamlit Cloud uses its Secrets settings for the API key. Your local `.env` file is not automatically uploaded to the cloud.

The ChromaDB knowledge base can be rebuilt from the documents when needed.

## 🧪 Testing

The project includes testing for document loading, retrieval, and agent behavior.

Important test scenarios include:

| Test Scenario | Expected Behavior |
|---|---|
| Relevant question | Retrieve relevant documents and generate an answer |
| Question requiring multiple documents | Retrieve supporting information from multiple sources |
| Initially irrelevant retrieval | Attempt corrective retrieval |
| Insufficient information | Return an insufficient-evidence response |
| Out-of-scope question | Avoid presenting unsupported information |
| Sensitive question | Flag for human review |

Initial project checks have confirmed that 10 documents can be loaded and indexed, the Groq connection works, and the deployed application can generate an answer.

Other test scenarios should be executed before being reported as passed.

## 🔐 Safety and Limitations

This application provides general educational information only.

- It does not diagnose mental health conditions.
- It does not prescribe medication.
- It does not provide personalized treatment plans.
- It is not a therapist or emergency service.
- Sensitive questions may be flagged for human review.
- Generated answers and automated validation can still contain errors.

If someone is experiencing a mental health emergency or is at risk of harming themselves or another person, they should seek immediate help from local emergency services or a qualified professional.

## 🚀 Future Improvements

- Add more credible mental health educational documents.
- Improve retrieval accuracy.
- Strengthen safety screening and human review.
- Improve source citation and evidence verification.
- Add more automated test cases.
- Improve the user interface.
- Support additional languages.

**Project:** Mental Health Knowledge and Psychoeducation Agent  
**Core Concepts:** RAG, LangGraph, LangChain, Embeddings, Vector Database, LLM Integration  
**LLM Provider:** Groq API  
**LLM Model:** GPT-OSS 20B  
**Interface:** Streamlit

*Built for educational and academic purposes.*﻿
