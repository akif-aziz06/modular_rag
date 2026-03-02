from llm_client import generate_response

def router(user_query: str):
    prompt = """
        You are an expert router for a Salesforce RAG system. 
        Your job is to route the query to the correct tool.

        ### TOOLS:
        1. vector_db: Use this for ALL Salesforce-related questions (e.g., Sales Cloud, Service Cloud, Apex, Objects, Modules).
        2. weather_api: Use this ONLY for current weather/temperature requests.
        3. web_search: Use this for general knowledge NOT related to Salesforce or weather.

        ### INSTRUCTIONS:
        - Return ONLY the tool name. 
        - Do not provide any explanation or quotes.
        - If the query mentions 'Salesforce', 'Cloud', 'Lead', 'Account', or any CRM term, use 'vector_db'.

        ### EXAMPLES:
        User query: "Hi What is sales cloud?"
        Router: vector_db
        
        User query: "What is the weather like in Karachi?"
        Router: weather_api
        
        User query: "Who won the World Cup in 2022?"
        Router: web_search
    """
    # Clean the output in case the LLM adds extra spaces or quotes
    return generate_response(user_query, prompt).strip().lower()

print(router("Hi WHICH TEAM REACHED IN SEMI-FINAL OF CRICKET T-20 WORLD CUP 2026?"))