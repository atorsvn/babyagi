import os, openai, pinecone, time, sys
from collections import deque
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY, OPENAI_API_MODEL, PINECONE_API_KEY, PINECONE_ENVIRONMENT, YOUR_TABLE_NAME, OBJECTIVE, YOUR_FIRST_TASK = [os.getenv(key) for key in ("OPENAI_API_KEY", "OPENAI_API_MODEL", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "TABLE_NAME", "OBJECTIVE", "FIRST_TASK")]
assert all(v for v in (OPENAI_API_KEY, OPENAI_API_MODEL, PINECONE_API_KEY, PINECONE_ENVIRONMENT, YOUR_TABLE_NAME, OBJECTIVE, YOUR_FIRST_TASK)), "Missing environment variable(s) from .env"

openai.api_key, pinecone.api_key, pinecone.environment = OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT
pinecone.create_index(YOUR_TABLE_NAME, dimension=1536, metric="cosine", pod_type="p1") if YOUR_TABLE_NAME not in pinecone.list_indexes() else None
index, task_list = pinecone.Index(YOUR_TABLE_NAME), deque([{"task_id": 1, "task_name": YOUR_FIRST_TASK}])

def openai_call(prompt: str, model: str = OPENAI_API_MODEL, temperature: float = 0.5, max_tokens: int = 100):
    api = openai.Completion.create if not model.startswith('gpt-') else openai.ChatCompletion.create
    messages = [{"role": "user", "content": prompt}] if api == openai.ChatCompletion.create else None
    response = api(engine=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, top_p=1, frequency_penalty=0, presence_penalty=0, messages=messages)
    return response.choices[0].message.content.strip() if api == openai.ChatCompletion.create else response.choices[0].text.strip()

def add_task(task: Dict): task_list.append(task)
def get_ada_embedding(text): return openai.Embedding.create(input=[text.replace("\n", " ")], model="text-embedding-ada-002")["data"][0]["embedding"]
def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]): return [{"task_name": task_name} for task_name in openai_call(f"You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array.").split('\n')]
def prioritization_agent(this_task_id: int): task_list.clear(); [task_list.append({"task_id": task_parts[0].strip(), "task_name": task_parts[1].strip()}) for task_string in openai_call(f"""You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {[t["task_name"] for t in task_list]}. Consider the ultimate objective of your team:{OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:\n#. First task\n#. Second task\nStart the task list with number {this_task_id + 1}.""").split('\n') if (task_parts := task_string.strip().split(".", 1)) and len(task_parts) == 2]

def execution_agent(objective: str, task: str) -> str:
    context = context_agent(query=objective, n=5)
    prompt = f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"
    return openai_call(prompt, temperature=0.7, max_tokens=2000)

def context_agent(query: str, n: int):
    query_embedding = get_ada_embedding(query)
    results = index.query(query_embedding, top_k=n, include_metadata=True)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [str(item.metadata['task']) for item in sorted_results]

task_id_counter = 1
while True:
    if task_list:
        task = task_list.popleft()
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])

        enriched_result = {'data': result}
        result_id = f"result_{task['task_id']}"
        vector = enriched_result['data']
        index.upsert([(result_id, get_ada_embedding(vector), {"task": task['task_name'], "result": result})])

    new_tasks = task_creation_agent(OBJECTIVE, enriched_result, task["task_name"], [t["task_name"] for t in task_list])

    for new_task in new_tasks:
        task_id_counter += 1
        new_task.update({"task_id": task_id_counter})
        add_task(new_task)

    prioritization_agent(this_task_id)
    time.sleep(1)
