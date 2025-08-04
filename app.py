import streamlit as st
import asyncio
import os
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

load_dotenv()

# Model Client
model_client = OpenAIChatCompletionClient(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

st.title("🧠Interview Simulator")

# Job position input
job_position = st.text_input("Enter Job Position:", "software_engineer")

# Start Interview
if st.button("🚀 Start Interview"):

    # Setup agents
    interviewer = AssistantAgent(
        name='interviewer',
        model_client=model_client,
        system_message=f'''
You are a professional interviewer for a {job_position} position.
Ask one clear question at a time and Wait for user to respond.
Ask 3 questions in total covering technical skills and experience, problem-solving abilities, and cultural fit.
After asking 3 questions, say "TERMINATE" at the end of the interview.
'''
    )

    candidate = UserProxyAgent(
        name='candidate',
        description=f"An agent that simulates candidate for a {job_position} position",
        input_func=input  # will override later in WebConsole
    )

    carrier_coach = AssistantAgent(
        name='carrier_coach',
        model_client=model_client,
        system_message=f'''
You are a career coach specializing in preparing candidates for {job_position} interviews.
Provide constructive feedback on the candidate's responses and suggest improvements.
After the interview, summarize the candidate's performance and provide actionable advice.
'''
    )

    # Group Chat Setup
    team = RoundRobinGroupChat(
        participants=[interviewer, candidate, carrier_coach],
        termination_condition=TextMentionTermination(text='TERMINATE'),
        max_turns=20
    )

    # Streaming Interface
    stream = team.run_stream(task=f"conducting an interview for a {job_position} position")

    async def run_web():
        await Console(stream)

    # Async Run
    asyncio.run(run_web())
