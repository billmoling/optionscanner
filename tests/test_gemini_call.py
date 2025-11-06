import os

from dotenv import load_dotenv

from google import genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment before running this script.")

client = genai.Client(api_key=API_KEY)
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Write a story about a magic backpack."
)
print(response.text)
