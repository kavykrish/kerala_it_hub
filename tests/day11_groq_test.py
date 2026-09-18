import os

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GET GROQ API KEY
# =========================================================

api_key = os.getenv(
    "GROQ_API_KEY"
)


if not api_key:

    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please check your .env file."
    )


# =========================================================
# CREATE GROQ CLIENT
# =========================================================

client = Groq(
    api_key=api_key
)


# =========================================================
# TEST GROQ
# =========================================================

response = client.chat.completions.create(

    model="openai/gpt-oss-20b",

    messages=[
        {
            "role": "user",
            "content": (
                "Say hello and explain in one sentence "
                "what a RAG system does."
            )
        }
    ],

    temperature=0
)


# =========================================================
# DISPLAY RESPONSE
# =========================================================

print("=" * 70)
print("KERALA IT HUB - DAY 11 GROQ TEST")
print("=" * 70)

print("\nGroq connection successful!")

print("\nModel response:")
print(
    response.choices[0].message.content
)

print("\n" + "=" * 70)
print("DAY 11 GROQ TEST COMPLETE")
print("=" * 70)