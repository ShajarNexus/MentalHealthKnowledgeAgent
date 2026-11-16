
import unittest

from unittest.mock import patch

from langchain_core.documents import Document

from rag_pipeline import assess_relevance

from agent import (
    safety_screen_node,
    human_review_node,
    insufficient_node
)


class TestMentalHealthAgent(unittest.TestCase):

    def test_relevant_retrieval(self):

        documents = [

            (
                Document(
                    page_content="Stress information",
                    metadata={"source": "stress.txt"}
                ),
                0.4
            )

        ]

        relevant, note = assess_relevance(documents)

        self.assertTrue(relevant)

    def test_irrelevant_retrieval(self):

        documents = [

            (
                Document(
                    page_content="Unrelated information",
                    metadata={"source": "other.txt"}
                ),
                2.0
            )

        ]

        relevant, note = assess_relevance(documents)

        self.assertFalse(relevant)

    def test_empty_retrieval(self):

        relevant, note = assess_relevance([])

        self.assertFalse(relevant)

    def test_insufficient_information(self):

        result = insufficient_node({

            "question": "Unknown question"

        })

        self.assertEqual(

            result["status"],

            "INSUFFICIENT EVIDENCE"

        )

    def test_sensitive_question(self):

        result = safety_screen_node({

            "question": "I want to hurt myself"

        })

        self.assertTrue(
            result["needs_human_review"]
        )

    def test_human_review_message(self):

        result = human_review_node({

            "question": "Sensitive question"

        })

        self.assertEqual(

            result["status"],

            "HUMAN REVIEW REQUIRED"

        )


if __name__ == "__main__":

    unittest.main()
