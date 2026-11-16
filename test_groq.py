
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API key from .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

# Connect to Groq
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Send a test question
response = llm.invoke(
    "Explain anxiety in two simple sentences."
)

print("\n========== GROQ TEST ==========")
print(response.content)
print("\nGroq connection successful!")
