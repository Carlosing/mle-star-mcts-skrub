"""Machine Learning Engineer: automate the implementation of ML models."""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file.
load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("OPENAI_API_BASE")
MODEL_NAME = os.environ.get("ROOT_AGENT_MODEL", "meta-llama-3.1-8b-instruct")

if not API_KEY or not API_BASE:
    raise ValueError(
        "CRITICAL: OPENAI_API_KEY or OPENAI_API_BASE not found. "
        "Make sure your .env file is configured."
    )

client = OpenAI(api_key=API_KEY, base_url=API_BASE)
