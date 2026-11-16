
import os
import re
from pathlib import Path
from typing import TypedDict

import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from config import TOP_K

from rag_pipeline import (
    retrieve_documents,
    assess_relevance
)

from prompts import (
    SYSTEM_PROMPT,
    REFLECTION_PROMPT
)


# =====================================================
# 1. LOAD LOCAL ENVIRONMENT VARIABLES
# =====================================================

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env"
)


# =====================================================
# 2. AGENT STATE
# =====================================================

class AgentState(TypedDict, total=False):

    question: str
    retrieved: list
    relevant: bool
    relevance_note: str
    correction_attempted: bool
    context: str
    sources: list
    draft_answer: str
    reflection: str
    final_answer: str
    needs_human_review: bool
    status: str


# =====================================================
# 3. GET GROQ API KEY
# =====================================================

def get_groq_api_key():

    # First check local .env or environment variables
    api_key = os.getenv("GROQ_API_KEY")

    # If not found, check Streamlit Cloud Secrets
    if not api_key:

        try:
            api_key = st.secrets.get("GROQ_API_KEY")

        except Exception:
            api_key = None

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please add it to your local .env file "
            "or Streamlit Community Cloud Secrets."
        )

    return api_key


# =====================================================
# 4. LOAD GROQ AI MODEL
# =====================================================

def get_model():

    api_key = get_groq_api_key()

    return ChatGroq(

        model="openai/gpt-oss-20b",

        api_key=api_key,

        temperature=0.1

    )


# =====================================================
# 5. SAFETY SCREENING NODE
# =====================================================

def safety_screen_node(state):

    question = state["question"].lower()

    sensitive_words = [

        "suicide",
        "kill myself",
        "self harm",
        "self-harm",
        "hurt myself",
        "end my life",
        "overdose",
        "abuse",
        "diagnose me",
        "which medication",
        "what medicine",
        "treatment for me"

    ]

    flagged = any(

        word in question

        for word in sensitive_words

    )

    return {

        "needs_human_review": flagged

    }


def initial_router(state):

    if state.get("needs_human_review"):

        return "human_review"

    return "retrieve"


# =====================================================
# 6. RETRIEVAL NODE
# =====================================================

def retrieve_node(state):

    results = retrieve_documents(

        state["question"],

        k=TOP_K

    )

    relevant, note = assess_relevance(results)

    return {

        "retrieved": results,

        "relevant": relevant,

        "relevance_note": note,

        "correction_attempted": False

    }


# =====================================================
# 7. RELEVANCE ROUTER
# =====================================================

def relevance_router(state):

    if state["relevant"]:

        return "prepare_context"

    if not state["correction_attempted"]:

        return "corrective_retrieval"

    return "insufficient"


# =====================================================
# 8. CORRECTIVE RETRIEVAL NODE
# =====================================================

def corrective_retrieval_node(state):

    revised_question = (

        "Mental health educational information about: "

        + state["question"]

    )

    results = retrieve_documents(

        revised_question,

        k=TOP_K + 2

    )

    relevant, note = assess_relevance(results)

    return {

        "retrieved": results,

        "relevant": relevant,

        "relevance_note": (

            "Corrective retrieval attempted. " + note

        ),

        "correction_attempted": True

    }


def corrective_router(state):

    if state["relevant"]:

        return "prepare_context"

    return "insufficient"


# =====================================================
# 9. PREPARE RETRIEVED CONTEXT
# =====================================================

def prepare_context_node(state):

    context_parts = []

    sources = []

    for document, distance in state["retrieved"]:

        source = document.metadata.get(

            "source",

            "Unknown"

        )

        context_parts.append(

            f"SOURCE: {source}\n"

            f"DISTANCE: {distance:.3f}\n"

            f"{document.page_content}"

        )

        if source not in sources:

            sources.append(source)

    return {

        "context": "\n\n".join(context_parts),

        "sources": sources

    }


# =====================================================
# 10. INSUFFICIENT EVIDENCE NODE
# =====================================================

def insufficient_node(state):

    return {

        "final_answer": (

            "I cannot provide a reliable answer from "

            "the available knowledge base. The retrieved "

            "information is insufficient or irrelevant. "

            "Please consult a qualified professional "

            "or an official health resource."

        ),

        "status": "INSUFFICIENT EVIDENCE",

        "reflection": (

            "Answer generation stopped because "

            "sufficient relevant evidence was unavailable."

        )

    }


# =====================================================
# 11. ANSWER GENERATION NODE
# =====================================================

def generate_node(state):

    messages = [

        SystemMessage(

            content=SYSTEM_PROMPT

        ),

        HumanMessage(

            content=(

                f"Question: {state['question']}\n\n"

                f"Retrieved evidence:\n"

                f"{state['context']}\n\n"

                "Generate an educational answer "

                "using only the retrieved evidence."

            )

        )

    ]

    response = get_model().invoke(messages)

    return {

        "draft_answer": response.content

    }


# =====================================================
# 12. REFLECTION AND VALIDATION NODE
# =====================================================

def reflect_node(state):

    messages = [

        SystemMessage(

            content=REFLECTION_PROMPT

        ),

        HumanMessage(

            content=(

                f"Retrieved context:\n"

                f"{state['context']}\n\n"

                f"Generated answer:\n"

                f"{state['draft_answer']}\n\n"

                "Start your response with exactly one verdict: "

                "SUPPORTED or NOT SUPPORTED."

            )

        )

    ]

    response = get_model().invoke(messages)

    reflection = str(response.content).strip()

    # Remove Markdown formatting and inspect first verdict
    normalized = re.sub(
        r"^[\s>*#`]+",
        "",
        reflection
    ).upper()

    first_word = re.match(
        r"^(SUPPORTED|NOT\s+SUPPORTED)\b",
        normalized
    )

    is_supported = (

        first_word is not None

        and first_word.group(1) == "SUPPORTED"

    )

    if is_supported:

        return {

            "reflection": reflection,

            "final_answer": state["draft_answer"],

            "status": "VALIDATED"

        }

    return {

        "reflection": reflection,

        "final_answer": (

            "The generated answer could not be "

            "validated against the retrieved sources. "

            "Therefore, it is not being presented "

            "as a reliable educational answer."

        ),

        "status": "NOT VALIDATED"

    }


# =====================================================
# 13. HUMAN REVIEW NODE
# =====================================================

def human_review_node(state):

    return {

        "needs_human_review": True,

        "final_answer": (

            "This question requires human review because "

            "it may involve a sensitive or potentially "

            "harmful situation. This AI assistant cannot "

            "assess personal risk or provide emergency care. "

            "If someone may be in immediate danger, contact "

            "local emergency services or a trusted person "

            "and seek qualified professional support. "

            "No personalized clinical guidance was generated."

        ),

        "status": "HUMAN REVIEW REQUIRED"

    }


# =====================================================
# 14. BUILD LANGGRAPH AGENT
# =====================================================

def build_agent():

    graph = StateGraph(AgentState)

    # Register nodes

    graph.add_node(
        "safety_screen",
        safety_screen_node
    )

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "corrective_retrieval",
        corrective_retrieval_node
    )

    graph.add_node(
        "prepare_context",
        prepare_context_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_node(
        "reflect",
        reflect_node
    )

    graph.add_node(
        "insufficient",
        insufficient_node
    )

    graph.add_node(
        "human_review",
        human_review_node
    )

    # Entry point

    graph.set_entry_point("safety_screen")

    # Safety routing

    graph.add_conditional_edges(

        "safety_screen",

        initial_router,

        {

            "human_review": "human_review",

            "retrieve": "retrieve"

        }

    )

    # Initial retrieval routing

    graph.add_conditional_edges(

        "retrieve",

        relevance_router,

        {

            "prepare_context": "prepare_context",

            "corrective_retrieval": "corrective_retrieval",

            "insufficient": "insufficient"

        }

    )

    # Corrective retrieval routing

    graph.add_conditional_edges(

        "corrective_retrieval",

        corrective_router,

        {

            "prepare_context": "prepare_context",

            "insufficient": "insufficient"

        }

    )

    # Answer generation workflow

    graph.add_edge(
        "prepare_context",
        "generate"
    )

    graph.add_edge(
        "generate",
        "reflect"
    )

    graph.add_edge(
        "reflect",
        END
    )

    # End nodes

    graph.add_edge(
        "human_review",
        END
    )

    graph.add_edge(
        "insufficient",
        END
    )

    return graph.compile()


# =====================================================
# 15. ASK AGENT
# =====================================================

def ask_agent(question):

    agent = build_agent()

    result = agent.invoke({

        "question": question

    })

    return result
