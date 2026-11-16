
import os
from pathlib import Path
from typing import TypedDict

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


# Load API key from .env file

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env"
)


# 1. Agent State

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


# 2. Load Groq AI Model

def get_model():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please check your .env file."
        )

    return ChatGroq(

        model="openai/gpt-oss-20b",

        api_key=api_key,

        temperature=0.1

    )


# 3. Safety Screening Node

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


# 4. Retrieval Node

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


# 5. Conditional Routing

def relevance_router(state):

    if state["relevant"]:

        return "prepare_context"

    if not state["correction_attempted"]:

        return "corrective_retrieval"

    return "insufficient"


# 6. Corrective Retrieval Node

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


# 7. Prepare Retrieved Context

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


# 8. Insufficient Evidence Node

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


# 9. Answer Generation Node

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


# 10. Reflection Node - Improved Validation

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

    # Normalize Markdown formatting before checking verdict

    normalized = reflection.upper()

    for character in [

        "*",

        "#",

        "`",

        ">",

        " ",

        "\n",

        "\t"

    ]:

        normalized = normalized.lstrip(character)

    # Accept only an explicit SUPPORTED verdict

    if (

        normalized.startswith("SUPPORTED")

        and not normalized.startswith("SUPPORTED NOT")

    ):

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


# 11. Human Review Node

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


# 12. Build LangGraph

def build_agent():

    graph = StateGraph(AgentState)

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

    graph.set_entry_point("safety_screen")

    graph.add_conditional_edges(

        "safety_screen",

        initial_router,

        {

            "human_review": "human_review",

            "retrieve": "retrieve"

        }

    )

    graph.add_conditional_edges(

        "retrieve",

        relevance_router,

        {

            "prepare_context": "prepare_context",

            "corrective_retrieval": "corrective_retrieval",

            "insufficient": "insufficient"

        }

    )

    graph.add_conditional_edges(

        "corrective_retrieval",

        corrective_router,

        {

            "prepare_context": "prepare_context",

            "insufficient": "insufficient"

        }

    )

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

    graph.add_edge(

        "human_review",

        END

    )

    graph.add_edge(

        "insufficient",

        END

    )

    return graph.compile()


# 13. Ask Question

def ask_agent(question):

    agent = build_agent()

    result = agent.invoke({

        "question": question

    })

    return result
