
import streamlit as st

from agent import ask_agent

from rag_pipeline import build_knowledge_base

from config import OLLAMA_MODEL, DATA_DIR


st.set_page_config(

    page_title="Mental Health Knowledge Agent",

    page_icon="🧠",

    layout="wide"

)


st.title(
    "🧠 Mental Health Knowledge and Psychoeducation Agent"
)

st.caption(
    "Evidence-grounded mental health educational assistant"
)


st.warning(

    "This application provides general educational "

    "information only. It is not a therapist or "

    "diagnostic system and does not provide personalized "

    "treatment instructions."

)


# Sidebar

with st.sidebar:

    st.header("Knowledge Base")

    st.write("Document folder:")

    st.code(str(DATA_DIR))

    st.write("AI Model:")

    st.write(OLLAMA_MODEL)

    if st.button("Build / Check Knowledge Base"):

        with st.spinner(
            "Processing mental health documents..."
        ):

            try:

                count = build_knowledge_base()

                st.success(

                    f"Knowledge base ready! "

                    f"{count} chunks indexed."

                )

            except Exception as error:

                st.error(str(error))


# Question Input

st.subheader("Ask Your Question")

question = st.text_area(

    "Enter a mental health education question:",

    placeholder="Example: What is anxiety?",

    height=120

)


if st.button(
    "Generate Educational Answer",
    type="primary"
):

    if not question.strip():

        st.error("Please enter a question.")

    else:

        with st.spinner(
            "The agent is processing your question..."
        ):

            try:

                result = ask_agent(
                    question.strip()
                )

                # Answer

                st.subheader("Educational Answer")

                st.markdown(
                    result.get(
                        "final_answer",
                        "No answer available."
                    )
                )

                # Relevance

                st.subheader(
                    "Retrieval Relevance Assessment"
                )

                st.write(

                    result.get(
                        "relevance_note",
                        "Not available"
                    )

                )

                # Sources

                st.subheader(
                    "Retrieved Document References"
                )

                sources = result.get(
                    "sources",
                    []
                )

                if sources:

                    for source in sources:

                        st.write(f"- {source}")

                else:

                    st.write(
                        "No accepted source references."
                    )

                # Reflection

                st.subheader(
                    "Reflection and Validation"
                )

                st.write(

                    result.get(
                        "reflection",
                        "Not available"
                    )

                )

                st.write(

                    "**Validation Status:**",

                    result.get(
                        "status",
                        "Unknown"
                    )

                )

                # Human review

                if result.get(
                    "needs_human_review"
                ):

                    st.error(
                        "Human review is required "
                        "for this sensitive question."
                    )

                # Retrieved evidence

                with st.expander(
                    "View Retrieved Evidence"
                ):

                    for document, distance in result.get(
                        "retrieved",
                        []
                    ):

                        st.write(

                            f"Source: "
                            f"{document.metadata.get('source')}"

                        )

                        st.write(
                            f"Distance: {distance:.3f}"
                        )

                        st.write(
                            document.page_content
                        )

                        st.divider()

            except Exception as error:

                st.error(
                    f"Error: {error}"
                )

st.divider()

st.caption(
    "For educational purposes only. "
    "Not a substitute for professional mental health care."
)
