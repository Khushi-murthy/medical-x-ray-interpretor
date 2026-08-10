import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")

print("API Key Loaded:", bool(api_key))

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents="Reply with only the word: SUCCESS"
    )

    print("SUCCESS")
    print(response.text)

except Exception as e:
    print(type(e).__name__)
    print(e)