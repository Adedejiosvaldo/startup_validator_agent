import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.tools import google_search





# root_agent = Agent(
#     name="basic_search_agent",
#     model="gemini-2.0-flash-exp",
#     description="A basic search agent that can answer questions using Google Search.",
#     tools=[google_search],
#     instruction="You are an expert researcher. You always stick to the facts and never make up information. If you don't know the answer, you will say 'I don't know'. You will always use Google Search to find the answer to the question.",
# )
