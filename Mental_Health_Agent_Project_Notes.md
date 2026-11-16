# Mental Health Knowledge and Psychoeducation Agent

## Project Notes in Easy English

> **Project purpose:** This application gives general, evidence-grounded
> mental health education using information stored in its own knowledge
> base. It is designed to explain topics and show source references. It
> is **not** a doctor, therapist, diagnostic system, or treatment
> service.

------------------------------------------------------------------------

## 1. What does the project do?

The project is a Mental Health Knowledge and Psychoeducation Agent.

A user enters a question, for example:

-   What is anxiety?
-   What is stress?
-   Why is sleep important?
-   What does emotional wellbeing mean?
-   When should someone consider professional support?

The system searches its local mental health documents, checks whether
the retrieved information is relevant, and then uses an AI language
model to prepare an educational response when enough supporting
information is available.

If the knowledge base does not contain enough useful information, the
system should say that it cannot provide a reliable answer from the
available sources. It should not invent an answer.

### Main objectives

1.  Answer general mental health education questions.
2.  Use trusted reference documents instead of depending only on the
    language model's memory.
3.  show the sources used for an answer.
4.  Identify when the available evidence is not sufficient.
5.  Avoid diagnosis, treatment instructions, and unsupported medical
    claims.

------------------------------------------------------------------------

## 2. Technologies used and why

  ------------------------------------------------------------------------
  Technology               What it does            Why we used it
  ------------------------ ----------------------- -----------------------
  **Python 3.11**          Main programming        It is used to write the
                           language                application's logic and
                                                   connect the different
                                                   libraries.

  **Streamlit**            User interface          It lets us build a
                                                   simple web app using
                                                   Python without building
                                                   a separate frontend in
                                                   React.

  **LangChain**            Connects components     It makes it easier to
                           such as document        build the RAG pipeline
                           loading, text           by connecting these
                           splitting, embeddings,  parts.
                           vector search, and the  
                           language model          

  **LangGraph**            Manages the agent's     It helps control which
                           workflow as connected   step runs next,
                           steps (nodes)           including decisions
                                                   such as whether
                                                   evidence is relevant or
                                                   another retrieval
                                                   attempt is needed.

  **Ollama**               Runs a language model   It allows the project
                           locally                 to use a local model
                                                   rather than requiring a
                                                   paid hosted AI API.

  **Qwen 2.5 3B**          Language model          It is used to generate
                                                   the natural-language
                                                   educational response
                                                   from the supplied
                                                   context.

  **Hugging Face Sentence  Creates numerical       It converts document
  Transformers**           representations called  text and questions into
                           embeddings              vectors so that
                                                   semantically similar
                                                   content can be found.

  **`all-MiniLM-L6-v2`**   Embedding model         It is the specific
                                                   embedding model
                                                   configured in this
                                                   project. It is
                                                   relatively small and
                                                   suitable for local
                                                   semantic search.

  **ChromaDB**             Vector database         It stores document
                                                   chunks and their
                                                   embeddings, and
                                                   searches for chunks
                                                   similar to a user's
                                                   question.

  **PyPDF**                Reads PDF documents     It allows the document
                                                   loader to extract text
                                                   from PDF files.

  **TXT files**            Store the project's     They provide the local
                           source material         mental health
                                                   information that the
                                                   RAG system searches.
  ------------------------------------------------------------------------

**Important difference:** Qwen generates the written response. Hugging
Face creates embeddings for searching. ChromaDB stores and retrieves
those embeddings. They do different jobs.

------------------------------------------------------------------------

## 3. What is RAG, and why did we use it?

**RAG** means **Retrieval-Augmented Generation**.

In easy words, RAG means:

1.  First, search for useful information in our own documents.
2.  Give the retrieved information to the language model.
3.  Ask the model to prepare an answer using that information.

### Why RAG is useful in this project

A normal language model may answer from information it learned during
training. It may also produce information that is not supported by our
project documents.

RAG helps ground the answer in the mental health references we provide.
It also makes it possible to show which documents were retrieved.

RAG does not guarantee that every answer is correct. That is why the
project also needs relevance checks, validation, and clear safety
limits.

------------------------------------------------------------------------

## 4. Our knowledge base

The project has a folder:

``` text
data/
└── mental_health/
```

It contains ten text documents:

1.  `anxiety.txt`
2.  `depression.txt`
3.  `emotional_wellbeing.txt`
4.  `mental_health_awareness.txt`
5.  `mental_health_terminology.txt`
6.  `professional_support.txt`
7.  `psychological_wellbeing.txt`
8.  `self_care.txt`
9.  `sleep.txt`
10. `stress.txt`

Each document contains information such as:

-   Title of the topic
-   Source organization
-   Reference URL
-   Short educational content

For example, `anxiety.txt` contains a title, the National Institute of
Mental Health as its source, a reference URL, and general information
about anxiety.

### Why did we create these documents?

They form the system's local knowledge base. When a user asks a
question, the RAG system searches these documents for relevant
information.

### Why include source names and URLs?

They help us identify where the educational information came from and
provide traceability. A source URL in a text file is a reference; it
does not mean the application automatically checks that website every
time a user asks a question.

------------------------------------------------------------------------

## 5. Document loading and text chunking

**Main file:** `document_loader.py`

This file has two important jobs.

### A. Loading documents

The `load_documents()` function:

-   Looks inside the mental health data folder.
-   Finds supported `.txt` and `.pdf` files.
-   Reads their text.
-   Creates LangChain `Document` objects.
-   Adds metadata such as the source filename and file path.

**Why do we use it?**

The system needs to read the source files before it can search them.

### B. Splitting documents into chunks

The `split_documents()` function uses LangChain's
`RecursiveCharacterTextSplitter`.

The current configuration is:

``` python
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
```

-   **Chunk size 700:** a document is divided into smaller text pieces,
    with a target size of about 700 characters.
-   **Overlap 100:** neighbouring chunks share about 100 characters.

**Why split documents?**

Searching one very large document can be less precise. Smaller chunks
help the system retrieve the part that is most related to the user's
question.

**Why use overlap?**

It helps avoid losing context when an important sentence or idea is near
a chunk boundary.

The loader test previously confirmed that the project loaded 10
documents and created 10 chunks with the current source files and
settings.

------------------------------------------------------------------------

## 6. Embeddings: what are they?

An **embedding** is a list of numbers that represents the meaning of a
piece of text.

For example, the question:

> What is anxiety?

and a document sentence about anxiety may use different words but have
similar meanings. Embeddings help the system find that relationship.

### Where do we use Hugging Face?

**Main file:** `rag_pipeline.py`

The project imports:

``` python
from langchain_huggingface import HuggingFaceEmbeddings
```

The embedding model is configured in `config.py`:

``` python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

The `get_embeddings()` function creates the embedding model:

``` python
_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Why use this model?

-   It creates embeddings for both document chunks and user questions.
-   It supports semantic search, which looks for meaning rather than
    only exact keyword matches.
-   The project can run it on the CPU.

**Remember:** Hugging Face embeddings do not write the final chatbot
answer. They help find relevant information.

------------------------------------------------------------------------

## 7. ChromaDB: why do we use it?

**Main file:** `rag_pipeline.py`

ChromaDB is our vector database.

In easy words, it stores the document chunks and their embeddings so the
system can search them later.

The project configuration includes:

``` python
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "mental_health_knowledge"
```

The `get_vectorstore()` function creates or opens the Chroma collection.

### What happens in ChromaDB?

1.  The document chunks are converted into embeddings.
2.  ChromaDB stores the chunks and their vector representations.
3.  A user asks a question.
4.  The question is converted into an embedding.
5.  ChromaDB finds chunks with similar meanings.
6.  The most relevant search results are returned to the application.

### Why not search only with normal keywords?

Keyword search may miss useful information when the user uses different
wording. Vector search can find text with similar meaning even if the
exact words are different.

------------------------------------------------------------------------

## 8. Building the knowledge base

**Main file:** `rag_pipeline.py`

The function `build_knowledge_base()` prepares the searchable
collection.

Its general process is:

1.  Open the Chroma vector store.
2.  Check whether it already contains document chunks.
3.  If data already exists, reuse it instead of adding the same
    documents again.
4.  If the collection is empty, load the source documents.
5.  Split the documents into chunks.
6.  Add the chunks to ChromaDB.

The function also supports `force_rebuild=True`, which deletes and
rebuilds the collection.

**Why check whether data already exists?**

So the app does not need to add the same documents every time it starts.

**Be careful:** Rebuilding deletes the existing collection before adding
the documents again. Use this when you intentionally want to refresh the
knowledge base.

------------------------------------------------------------------------

## 9. Retrieval and relevance checking

**Main file:** `rag_pipeline.py`

### Retrieval

The function:

``` python
retrieve_documents(question, k=TOP_K)
```

searches ChromaDB for chunks related to the user's question.

The current setting is:

``` python
TOP_K = 5
```

This means the system requests up to five matching chunks.

### Relevance assessment

The function:

``` python
assess_relevance(results)
```

checks the distances returned by ChromaDB.

The current configuration is:

``` python
MAX_RELEVANCE_DISTANCE = 1.25
```

The function finds the best (lowest) distance. If it is at or below the
configured threshold, the results are treated as relevant.

**Easy meaning of distance:**

-   A lower distance generally means the retrieved text is closer to the
    question in the vector space.
-   A higher distance generally means it is less similar.
-   The threshold is a project setting, not a universal medical or
    scientific rule.

**Important limitation:** The current `assess_relevance()` function
checks the best distance. It does not mean that every returned chunk is
relevant. A result list can contain some useful chunks and some weak
matches.

### Example from our test

For the test question **"What is anxiety?"**, the project retrieved five
chunks. The `anxiety.txt` result had a distance of about `0.504`, and
the relevance check returned `True` because this was below the
configured threshold of `1.25`.

This test showed that the document loading, embeddings, ChromaDB
storage, retrieval, and basic relevance check were working.

------------------------------------------------------------------------

## 10. Language model: Ollama and Qwen

The project configuration contains:

``` python
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
```

### What is Ollama?

Ollama is a tool that runs supported AI language models on the local
computer.

### What is Qwen 2.5 3B?

Qwen 2.5 3B is the language model selected for this project. The model
is used to turn the retrieved information into a readable educational
response.

### Why use a local model?

-   It avoids depending on a paid hosted model API for normal
    generation.
-   It can run locally after the model has been downloaded.
-   It is useful for a student project with a limited budget.

### What does the model receive?

In a RAG workflow, the model should receive the user's question and the
retrieved source information. The prompt should instruct it to stay
within the evidence and say when the evidence is insufficient.

------------------------------------------------------------------------

## 11. LangChain and LangGraph: what is the difference?

These tools are related, but they are not the same.

### LangChain

LangChain helps connect components, for example:

-   Document objects
-   Text splitting
-   Embedding models
-   Vector databases
-   Language models

In our project, the document loader and RAG pipeline use LangChain
components.

### LangGraph

LangGraph is used to organize an agent workflow into steps called
**nodes**.

A node performs one job. The workflow can move from one node to another
depending on the result of a step.

For this project, the intended workflow includes steps such as:

-   Receive the user's question.
-   Retrieve relevant documents.
-   Assess whether the retrieved evidence is relevant.
-   Try corrective retrieval when the first search is not good enough,
    if that path is implemented.
-   Generate an educational response when enough evidence is available.
-   Reflect on or validate the response.
-   Stop or report insufficient evidence when the sources do not support
    a reliable answer.

**Why use LangGraph?**

A simple RAG pipeline is usually a straight sequence. An agent workflow
may need decisions and different paths. LangGraph helps represent and
control those paths.

**Important:** The exact node names, edges, conditions, and retry limits
must be checked in the project's `agent.py`. This guide describes the
intended workflow; use the actual source code when explaining exact
implementation details to a teacher.

------------------------------------------------------------------------

## 12. Streamlit: the user interface

**Main file:** `app.py`

Streamlit provides the page where a user interacts with the project.

The general user journey is:

1.  Open the Streamlit application.
2.  Enter a mental health education question.
3.  Submit the question.
4.  The backend processes it using the agent and RAG components.
5.  The page displays the response and, where available, source
    references and validation information.

### Why Streamlit?

-   It is Python-based.
-   It is quick to build and demonstrate.
-   It avoids needing a separate React or mobile frontend for this
    project.

------------------------------------------------------------------------

## 13. Overall system workflow

``` text
Trusted mental health TXT/PDF documents
                 |
                 v
        document_loader.py
       Read and load documents
                 |
                 v
        Text chunking
       (size 700, overlap 100)
                 |
                 v
     Hugging Face embeddings
       all-MiniLM-L6-v2
                 |
                 v
            ChromaDB
       Store searchable chunks
                 |
                 |
User question --> Streamlit app
                 |
                 v
           Agent workflow
             LangGraph
                 |
                 v
        Retrieve relevant chunks
                 |
                 v
       Check retrieval relevance
                 |
          +------+------+
          |             |
       Relevant      Not relevant
          |             |
          v             v
   Qwen via Ollama   Corrective search /
   prepares answer   insufficient evidence
          |
          v
     Validate response
          |
          v
   Show educational answer,
   source references and status
```

This is a high-level explanation. The exact paths and node connections
should match the current `agent.py` implementation.

------------------------------------------------------------------------

## 14. Safety rules

Mental health is a sensitive subject, so the application must follow
clear limits.

1.  **Education only:** Explain general concepts; do not act as a mental
    health professional.
2.  **No diagnosis:** Do not tell a user that they have depression,
    anxiety, or another condition.
3.  **No treatment decisions:** Do not prescribe medication or create a
    personal treatment plan.
4.  **Use evidence:** Base answers on the retrieved project sources.
5.  **Do not invent facts:** If evidence is missing, clearly say so.
6.  **Show references:** Display the source documents used when
    possible.
7.  **Sensitive or harmful situations:** Follow the project's safety and
    human-review process. Do not treat an automated answer as a
    substitute for urgent professional or emergency help.

A disclaimer alone is not enough. The system's prompts, workflow,
validation, and user interface should all support these limits.

------------------------------------------------------------------------

## 15. Important configuration settings

**Main file:** `config.py`

  ---------------------------------------------------------------------------------------------
  Setting                    Current value                              Meaning
  -------------------------- ------------------------------------------ -----------------------
  `DATA_DIR`                 `data/mental_health`                       Location of source
                                                                        documents

  `CHROMA_DIR`               `chroma_db`                                Local ChromaDB storage
                                                                        folder

  `OLLAMA_MODEL`             `qwen2.5:3b`                               Language model name

  `OLLAMA_BASE_URL`          `http://localhost:11434`                   Local Ollama service
                                                                        address

  `EMBEDDING_MODEL`          `sentence-transformers/all-MiniLM-L6-v2`   Model used to create
                                                                        embeddings

  `COLLECTION_NAME`          `mental_health_knowledge`                  ChromaDB collection
                                                                        name

  `CHUNK_SIZE`               `700`                                      Target chunk size

  `CHUNK_OVERLAP`            `100`                                      Shared text between
                                                                        neighbouring chunks

  `TOP_K`                    `5`                                        Number of chunks
                                                                        requested during
                                                                        retrieval

  `MAX_RELEVANCE_DISTANCE`   `1.25`                                     Configured
                                                                        relevance-distance
                                                                        threshold
  ---------------------------------------------------------------------------------------------

Changing a setting can change retrieval behaviour. For example,
increasing `TOP_K` retrieves more chunks, but some extra chunks may be
less relevant. Changing the relevance threshold also changes which
results pass the relevance check.

------------------------------------------------------------------------

## 16. Important project files

  -----------------------------------------------------------------------
  File or folder                      Responsibility
  ----------------------------------- -----------------------------------
  `app.py`                            Streamlit user interface and app
                                      entry point

  `config.py`                         Central project settings

  `document_loader.py`                Loads TXT/PDF files and splits them
                                      into chunks

  `rag_pipeline.py`                   Creates embeddings, opens ChromaDB,
                                      builds the knowledge base,
                                      retrieves documents, and assesses
                                      relevance

  `agent.py`                          Agent workflow and its
                                      decision-making logic; check this
                                      file for exact LangGraph nodes and
                                      edges

  `data/mental_health/`               Source knowledge documents

  `chroma_db/`                        Local vector database storage

  `test_loader.py`                    Tests document loading and chunking

  `test_rag.py`                       Tests knowledge-base creation,
                                      vector retrieval, and relevance
                                      assessment

  `requirements.txt`                  Python packages required by the
                                      project

  `.venv/`                            Local Python virtual environment;
                                      normally not shared as project
                                      source code
  -----------------------------------------------------------------------

If your current project has additional files, explain them according to
their actual code. Do not claim a function exists in a file until you
have checked it.

------------------------------------------------------------------------

## 17. How to run the project

Run these commands in the VS Code terminal from the project folder.

### Step 1: Activate the virtual environment

PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

If your environment folder is named `venv` instead, use:

``` powershell
.\venv\Scripts\Activate.ps1
```

### Step 2: Install the packages

``` powershell
pip install -r requirements.txt
```

### Step 3: Start Ollama

Make sure Ollama is installed and running. Download the configured model
if it is not already available:

``` powershell
ollama pull qwen2.5:3b
```

### Step 4: Start the Streamlit application

``` powershell
streamlit run app.py
```

The terminal will show the local address for the application. Open that
address in your browser.

------------------------------------------------------------------------

## 18. Tests already performed

The following checks were reported during project development:

### Document loader test

-   10 source documents were loaded.
-   10 chunks were created with the current documents and settings.

### RAG test

-   ChromaDB initially contained 0 chunks.
-   The knowledge base was built.
-   10 chunks were stored.
-   The question `What is anxiety?` retrieved 5 results.
-   The best result was `anxiety.txt`, with a distance of approximately
    `0.504`.
-   The relevance function returned `True` using the configured
    threshold of `1.25`.

These tests confirm that the loader and basic RAG retrieval pipeline
worked in that test run. They do not, by themselves, prove that the
complete Streamlit and LangGraph workflow is correct or that every
generated answer is clinically accurate.

------------------------------------------------------------------------

## 19. Test scenarios to demonstrate

Use these kinds of questions when testing the complete application:

  -----------------------------------------------------------------------
  Test                    Example question        What should happen
  ----------------------- ----------------------- -----------------------
  Relevant question       What is anxiety?        Retrieve relevant
                                                  anxiety information and
                                                  answer from the
                                                  sources.

  Multiple-topic question How are stress and      Retrieve useful
                          sleep connected?        information from more
                                                  than one topic if the
                                                  sources support it.

  Weak or irrelevant      Explain how to repair a Do not present
  match                   car engine.             unrelated mental health
                                                  text as evidence.

  Insufficient evidence   Ask for a very specific Clearly state that the
                          fact that is not in the knowledge base does not
                          documents.              contain enough
                                                  evidence.

  Out-of-scope question   What is the current     Explain that the
                          price of a phone?       question is outside the
                                                  agent's mental health
                                                  education scope.

  Safety-sensitive        Ask the system to       Do not diagnose;
  question                diagnose a mental       explain the system's
                          health condition.       limits and direct the
                                                  user toward qualified
                                                  support.
  -----------------------------------------------------------------------

Record the actual result for each test during your demonstration. Do not
claim that a test passed until you have run it.

------------------------------------------------------------------------

## 20. Common viva questions and easy answers

### Q1. What is the main purpose of your project?

It is an evidence-grounded mental health education agent. It searches
trusted documents and provides general educational information with
source references.

### Q2. Why did you use RAG?

We used RAG so the model can use information retrieved from our own
knowledge base instead of relying only on its training knowledge.

### Q3. What is an embedding?

An embedding is a numerical representation of text. It helps the system
compare the meaning of a question with the meaning of document chunks.

### Q4. Why did you use Hugging Face?

We used Hugging Face's `all-MiniLM-L6-v2` model to convert questions and
document chunks into embeddings for semantic search.

### Q5. Why did you use ChromaDB?

ChromaDB stores the chunks and their embeddings and retrieves chunks
that are similar to the user's question.

### Q6. Why did you use LangChain?

LangChain provides components that help us load documents, split text,
create embeddings, connect the vector store, and work with language
models.

### Q7. Why did you use LangGraph?

LangGraph organizes the agent into steps and supports conditional paths,
such as continuing when evidence is relevant or taking another path when
it is not.

### Q8. What is the role of Ollama?

Ollama runs the selected language model locally on our computer.

### Q9. What is the role of Qwen?

Qwen generates the natural-language educational response using the
question and the retrieved context supplied by the application.

### Q10. What is the role of Streamlit?

Streamlit provides the user interface where users enter questions and
view responses.

### Q11. What is chunking?

Chunking means dividing long documents into smaller pieces so the system
can search for the most useful sections.

### Q12. Why do we use chunk overlap?

Overlap keeps some text shared between neighbouring chunks. It helps
preserve context around chunk boundaries.

### Q13. What does TOP_K = 5 mean?

The retrieval function requests up to five matching chunks from
ChromaDB.

### Q14. What is the relevance threshold?

It is a configured distance limit used by our relevance function to
decide whether the best retrieved result is close enough to the
question. It is a project heuristic, not a universal standard.

### Q15. What happens if the system cannot find enough evidence?

It should say that the available knowledge base does not contain enough
information for a reliable answer rather than inventing one.

### Q16. Can this project diagnose depression?

No. It is an educational system, not a diagnostic tool or a replacement
for a qualified professional.

### Q17. What is the difference between LangChain and LangGraph?

LangChain supplies useful building blocks for LLM applications.
LangGraph controls a workflow made of nodes, edges, and conditional
decisions.

### Q18. Does Hugging Face generate the answer?

No. In this project Hugging Face creates embeddings for search. Qwen is
the language model used for response generation.

### Q19. Does ChromaDB generate answers?

No. ChromaDB stores and retrieves relevant document chunks. The language
model generates the written response.

### Q20. Does RAG guarantee that an answer is correct?

No. RAG helps ground an answer in retrieved sources, but retrieval and
generation can still make mistakes. We need validation, source checks,
and safety controls.

------------------------------------------------------------------------

## 21. A short explanation to tell your teacher

"Our project is a Mental Health Knowledge and Psychoeducation Agent. We
created ten local mental health reference documents. First, our document
loader reads the TXT and PDF files and divides their text into smaller
chunks. Hugging Face's `all-MiniLM-L6-v2` model converts those chunks
into embeddings, and ChromaDB stores them for semantic search. When a
user asks a question, the RAG pipeline retrieves the most similar chunks
and checks their relevance using a configured distance threshold. The
agent workflow is organized with LangGraph, while Qwen 2.5 3B running
through Ollama is used to generate an educational response from the
retrieved context. Streamlit provides the user interface. The system is
intended for education only: it must not diagnose, prescribe treatment,
or invent information when the sources are insufficient."

------------------------------------------------------------------------

## 22. Important points to remember

-   **Streamlit** = user interface.
-   **Python** = main programming language.
-   **LangChain** = connects the RAG components.
-   **LangGraph** = controls the agent workflow and decisions.
-   **Hugging Face** = creates embeddings.
-   **`all-MiniLM-L6-v2`** = the selected embedding model.
-   **ChromaDB** = stores and searches vectors and document chunks.
-   **Ollama** = runs the language model locally.
-   **Qwen 2.5 3B** = generates the written response.
-   **RAG** = retrieve information first, then generate an answer using
    it.
-   **Chunking** = divide long documents into smaller pieces.
-   **Relevance check** = estimate whether retrieved chunks match the
    question.
-   **Validation** = check whether the response follows the available
    evidence and project rules.
-   **Safety** = education only, no diagnosis, no treatment, no
    unsupported claims.

**Final reminder:** Before a viva or demonstration, open your current
`agent.py` and `app.py` and verify the exact node names, conditional
routes, retry behaviour, validation logic, and displayed fields. The
exact implementation must always be explained from the code that is
actually in your project.
