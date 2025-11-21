from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import scrape_menu

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

query = input("I will find you the cheapest meal.\nPlease paste a URL to the restaurant menu: ")

system_prompt = """
You are a price comparison assistant.

I need you to scrape a restaurant webpage (that the user provides a URL to) using the `scrape_menu` tool. 

Given a user request, you may:
- call scrape_menu(url) to get the entire menu from the URL.
- From the returned response of scrape_menu, analyze all the items and prices.
- Determine the cheapest main dish + drink combination available. An appetizer or kids menu item does not count as a main dish.
- Return to the user the name of the restaurant, the cheapest main dish + drink combination and the total price in this format: 
  restaurant name: main dish ($price) + drink ($price) -> Total: $total_price
- If a drink is included with the main dish, mention it in your response, e.g. "main dish (includes drink)".
"""

tools = [scrape_menu]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

result_state = agent.invoke(
    {
        "messages": [
            {"role": "user", "content" : query}
        ]
    }
)

returned_response = result_state["messages"][-1].content
if isinstance(returned_response, list):
    returned_response = "".join([x.get("text", "") if isinstance(x, dict) else str(x) for x in returned_response])

print("\n---- Agent Response ----")
print(returned_response)


