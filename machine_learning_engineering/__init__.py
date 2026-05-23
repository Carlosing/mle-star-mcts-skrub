import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the hidden .env file
load_dotenv()

# Get the university credentials
API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("OPENAI_API_BASE")
MODEL_NAME = os.environ.get("ROOT_AGENT_MODEL", "meta-llama-3.1-8b-instruct")

if not API_KEY or not API_BASE:
    raise ValueError(
        "CRITICAL: OPENAI_API_KEY or OPENAI_API_BASE not found in the environment. "
        "Make sure your .env file is configured."
    )

# Initialize the global client used by the project
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

print(f"\n🚀 [SYSTEM] Engine initialized successfully.")
print(f"📡 [SYSTEM] Connected to the University's Proxy (SAIA).")
print(f"🧠 [SYSTEM] Model loaded: {MODEL_NAME}\n")