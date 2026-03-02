from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def generate_response(user_query: str, prompt: str):
    # Changed from responses.create to chat.completions.create
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ],
        model="llama-3.1-8b-instant", # Ensure you use a valid Groq model ID
        max_tokens=200,         # Changed from tokens=200
    )
    # Correct way to access the string content
    return response.choices[0].message.content

# print(generate_response("Hi How are you", "You are a helpful assistant."))