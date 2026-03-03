from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def generate_response(user_query: str, prompt: str, chat_history: list = None) -> str:
    """
    Generate a response from the Groq LLM.

    Args:
        user_query   : The current user message.
        prompt       : The system-level instruction prompt.
        chat_history : Optional list of previous messages in the format
                       [{"role": "user"|"assistant", "content": "..."}]
                       Injected between the system prompt and the current query.
    Returns:
        The LLM's response as a plain string.
    """
    messages = [{"role": "system", "content": prompt}]

    # Inject conversation history (if any) before the current query
    if chat_history:
        messages.extend(chat_history)

    # Append the current user query at the end
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.1-8b-instant",
        max_tokens=512,
    )
    return response.choices[0].message.content