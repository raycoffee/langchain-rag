from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from match_query import search_similar_chunks
import os
import json

load_dotenv()

def autonomous_search(query: str, max_attempts: int = 3):

    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    
    def evaluate_results(query: str, context: str):

        evaluation_prompt = f"""Given this query: "{query}"
        And this context: "{context}"
        
        Evaluate if the context contains the answer to the query. Think step by step:
        1. What specific information is the query asking for?
        2. Is that information present in the context?
        3. If not, what's missing?
        4. How could we rephrase the query to better find the information? Look at the style of writing in the context. Remember, we are doing cosine similarity search, via vector embeddings. Hence, if the query question did not have relevant answer in the context, we need to rephrase the query question to have a better chance of fetching the correct answer.
        
        Return your analysis as JSON with these field. Please respond only in valid JSON format with this structure:
        - has_answer: boolean
        - reasoning: string
        - suggested_query: string (if has_answer is false)"""

        messages = [HumanMessage(content=evaluation_prompt)]
        response = llm.invoke(messages)

        
        
        try:
            return json.loads(response.content)
        except:
            return {"has_answer": False, "reasoning": "Failed to parse response", "suggested_query": query}

    # Initial search
    results = search_similar_chunks(query)
    context = "\n\n".join([result['content'] for result in results])
    
    for attempt in range(max_attempts):
        print(f"🔍 Search attempt {attempt + 1}...")
        
        evaluation = evaluate_results(query, context)
        print("🔥", evaluation, "🔥")
        
        if evaluation["has_answer"]:
            return {
                "context": context,
                "results": results,
                "attempts": attempt + 1,
                "final_query": query,
                "reasoning": evaluation["reasoning"]
            }
        
        # If no answer found, try with the suggested query
        query = evaluation["suggested_query"]
        print(f"📝 Trying refined query: {query}")
        
        results = search_similar_chunks(query)
        context = "\n\n".join([result['content'] for result in results])
    
    return {
        "context": context,
        "results": results,
        "attempts": max_attempts,
        "final_query": query,
        "reasoning": evaluation["reasoning"]
    }

def generate_response(query: str) -> str:
    """
    Generate a response using relevant context from vector search and Claude
    
    Args:
        query (str): The user's question
        
    Returns:
        str: Claude's response based on the relevant context
    """
    print("🔍 Starting autonomous search...")
    search_results = autonomous_search(query)
    
    print(f"📚 Found context after {search_results['attempts']} attempts")
    print(f"Final query used: {search_results['final_query']}")
    
    # Initialize Claude
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    
    # Create the prompt with context and search process
    prompt = f"""Here is some relevant context from a document:

{search_results['context']}

Search process:
- Original query: {query}
- Final query used: {search_results['final_query']}
- Number of search attempts: {search_results['attempts']}
- Reasoning: {search_results['reasoning']}

Based on this context and search process, please answer the original question:
{query}

If you cannot find the answer in the context, explain what was tried and why the information couldn't be found."""

    # Get response from Claude
    messages = [
        SystemMessage(content="You are a helpful assistant answering questions about a book. Be explicit about your reasoning and what information you found or couldn't find."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    
    return response.content

print("🔍 Searching for relevant context...", generate_response("How does alice meet the mad hatter?"))