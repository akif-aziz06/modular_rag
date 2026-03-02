from llm_client import generate_response

def router(user_query: str):
    ROUTER_PROMPT = """
    You are a Salesforce Education Router. Classify the query into one of these 6 modules:

    1. admin_mode: Setup, configuration, and declarative (no-code) features.
       Examples: "How do I create a Custom Object?" or "Where is the Password Policy setting?"

    2. dev_mode: Apex, LWC, Triggers, SOQL, and technical implementation.
       Examples: "Show me a sample LWC component" or "How to handle governor limits in a loop?"

    3. consultant_mode: High-level strategy, architecture, and business "best-fit" solutions.
       Examples: "When should I use a Big Object over a Standard Object?" or "Compare Flow vs. Workflow."

    4. interview_mode: Preparing for Salesforce jobs, certifications, and recruiter-ready answers.
       Examples: "What are the common questions for Admin Cert?" or "How do I explain 'Role Hierarchy' in an interview?"

    5. interactive_mode: Requesting a step-by-step tutorial, a quiz, or a deep-dive conversation.
       Examples: "Quiz me on Salesforce Security" or "Walk me through the Lead conversion process step-by-step."

    6. general_queries: For anything NOT related to Salesforce (Sports, Weather, General Chat).
       Examples: "Who won the world cup?" or "Tell me a joke."

    INSTRUCTION: Return ONLY the module name (e.g., dev_mode). No punctuation or extra words.
    """
    return generate_response(user_query, ROUTER_PROMPT).strip().lower()

# This will now correctly return "general_queries" instead of guessing a Salesforce tool
print(router("Hi WHICH TEAM REACHED IN SEMI-FINAL OF CRICKET T-20 WORLD CUP 2026?"))