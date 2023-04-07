import os
import openai
import pinecone
import time
import sys
from collections import deque
from typing import Dict, List
from dotenv import load_dotenv
import os
import discord
from discord.ext import commands

load_dotenv()

# Set Discord bot token
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
assert DISCORD_BOT_TOKEN, "DISCORD_BOT_TOKEN environment variable is missing from .env"

# Set API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(f"\033[91m\033[1m"+"\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"+"\033[0m\033[0m")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
assert PINECONE_API_KEY, "PINECONE_API_KEY environment variable is missing from .env"

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
assert PINECONE_ENVIRONMENT, "PINECONE_ENVIRONMENT environment variable is missing from .env"

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Project config
OBJECTIVE = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OBJECTIVE", "")
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"

YOUR_FIRST_TASK = os.getenv("FIRST_TASK", "")
assert YOUR_FIRST_TASK, "FIRST_TASK environment variable is missing from .env"

# Configure OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create Pinecone index
table_name = YOUR_TABLE_NAME
dimension = 1536
metric = "cosine"
pod_type = "p1"
if table_name not in pinecone.list_indexes():
    pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)

# Connect to the index
index = pinecone.Index(table_name)

# Task list
task_list = deque([])

# Functions
def add_task(task: Dict):
    task_list.append(task)

def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

def openai_call(prompt: str, model: str = OPENAI_API_MODEL, temperature: float = 0.5, max_tokens: int = 100):
    if not model.startswith('gpt-'):
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()
    else:
        messages=[{"role": "user", "content": prompt}]
        response = openai.ChatCompletion
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()

def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]):
    prompt = f"You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array."
    response = openai_call(prompt)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks]

def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id)+1
    prompt = f"""You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team:{OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt)
    new_tasks = response.split('\n')
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})

def execution_agent(objective: str, task: str) -> str:
    context=context_agent(query=objective, n=5)
    prompt =f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"
    return openai_call(prompt, temperature=0.7, max_tokens=2000)

def context_agent(query: str, n: int):
    query_embedding = get_ada_embedding(query)
    results = index.query(query_embedding, top_k=n, include_metadata=True)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)    
    return [(str(item.metadata['task'])) for item in sorted_results]

async def send_large_message(ctx, content):
    if len(content) <= 2000:
        await ctx.send(content)
    else:
        parts = [content[i:i+2000] for i in range(0, len(content), 2000)]
        for part in parts:
            await ctx.send(part)

# Add the first task
first_task = {
    "task_id": 1,
    "task_name": YOUR_FIRST_TASK
}

add_task(first_task)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"{bot.user.name} has connected to Discord!")

@bot.command()
async def start(ctx):
    global task_list
    task_id_counter = 1
    while True:
        if task_list:
            # Print the task list
            await ctx.send("**Task List:**")
            for t in task_list:
                await ctx.send(str(t['task_id'])+": "+t['task_name'])

            # Step 1: Pull the first task
            task = task_list.popleft()
            await ctx.send("**Next Task:**")
           
            await ctx.send(str(task['task_id'])+": "+task['task_name'])

            # Step 2: Execute the task
            objective = OBJECTIVE
            result = execution_agent(objective, task['task_name'])
            await ctx.send("**Task Result:**")
            await send_large_message(ctx, result)


            # Step 3: Generate new tasks based on the result
            new_tasks = task_creation_agent(objective, result, task['task_name'], [t['task_name'] for t in task_list])
            for new_task in new_tasks:
                task_id_counter += 1
                new_task["task_id"] = task_id_counter
                add_task(new_task)

            # Step 4: Prioritize the tasks
            prioritization_agent(task["task_id"])

        else:
            await ctx.send("Task list is empty.")
            break

@bot.command()
async def stop(ctx):
    await ctx.send("Stopping bot...")
    await bot.logout()

bot.run(DISCORD_BOT_TOKEN)
