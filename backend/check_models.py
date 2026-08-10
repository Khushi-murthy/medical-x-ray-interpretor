import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available Models:\n")

for model in client.models.list():
    print(model.name)