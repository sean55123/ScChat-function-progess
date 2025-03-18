import openai
from functions.Tools_reasoning import function_descriptions
import pickle
import json
import os
import pandas as pd
import scanpy as sc
import json

openai.api_key = ""
with open("test_summary.pkl", "rb") as file:
    final_summary = pd.read_pickle(file)

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": final_summary},
    ],
)
agent_message1 = response.choices[0].message.content
print("First round annotation finished.")

results = {
    "Raw data": final_summary,
    "agent_message 1": agent_message1
}

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": final_summary},
    ],
)
agent_message2 = response.choices[0].message.content
print("Second round annotation finished.")

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": final_summary},
    ],
)
agent_message3 = response.choices[0].message.content
print("Third round annotation finished.")

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": final_summary},
    ],
)
agent_message4 = response.choices[0].message.content
print("Fourth round annotation finished.")

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": final_summary},
    ],
)
agent_message5 = response.choices[0].message.content
print("Fifth round annotation finished.")

ensembled_results = {
    "Raw data": final_summary,
    "response 1": agent_message1,
    "response 2": agent_message2,
    "response 3": agent_message3,
    "response 4": agent_message4,
    "response 5": agent_message5
}

supervisor_content = """
You are a supervisor agent tasked with consolidating the outputs from multiple annotation trials.
Each previous trial has analyzed clustering statistics and provided classification labels for each cluster based on a given reference.
Your goal is to review these outputs and decide on the most likely and accurate annotation for each cluster.

Please consider the following:
- Clearly identify any discrepancies or commonalities among the trials.
- Explain your reasoning for selecting the final annotation.
- Format your final output as a summary that lists each cluster along with its consolidated annotation and a brief rationale.

Remember to base your decision on the clustering statistics provided in the reference and the outputs from the previous trials.
*** The output should stay in the format like this: group_to_cell_type = {'0': 'Myeloid cells','1': 'T cells','2': 'Myeloid cells','3': 'Myeloid cells','4': 'T cells'} without further explanation or comment.
"""
response = openai.chat.completions.create(
    model="o3-mini",
    messages=[
        {"role": "system", "content": supervisor_content},
        {"role": "user", "content": json.dumps(ensembled_results)},
    ],
)
agent_message = response.choices[0].message.content
print(agent_message)