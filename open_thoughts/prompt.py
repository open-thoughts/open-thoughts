import json

from datasets import Dataset

DEEPSEEK_R1_SYSTEM_PROMPT = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered
thinking process.
"""

SKY_T1_SYSTEM_PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"  # noqa


def format_code_prompt(x):
    formatted_prompt = ""

    data = json.loads(x["test_cases"])
    if not data.get("fn_name"):
        formatted_prompt += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # noqa
    else:
        formatted_prompt += (
            "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # noqa
        )

    formatted_prompt += x["problem"]
    if x["starter_code"] is not None:
        data = x["starter_code"]
        data = "\n" + data
        formatted_prompt += data
    return formatted_prompt


def map_to_share_gpt(x):
    if x["domain"] == "code" and "formatted_prompt" not in x:
        user = format_code_prompt(x)
    elif x["domain"] == "math":
        user = f"Return your final response within \\boxed{{}}. {x['question']}"
    else:
        user = x["question"]

    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user},
            {
                "from": "assistant",
                "value": f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>",
            },
        ],
    }


def map_numina_conversations(x):
    """Map the Numina dataset to the required format."""
    user_message = f"Return your final response within \\boxed{{}}. {x['problem']}"
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )
    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


def apply_numina_map(dataset: Dataset) -> Dataset:
    numina_conversations = dataset.map(map_numina_conversations)
    return numina_conversations


def map_apps_conversations(x):
    """Map the APPS dataset to the required format."""
    test_case = json.loads(x["input_output"])
    starter_code = x["starter_code"]
    prompt = x["question"]

    user_message = ""
    data = test_case
    if not data.get("fn_name"):
        user_message += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # "\nUse Standard Input format"#\n" #noqa
    else:
        user_message += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # "\nUse Call-Based format"#\n" #noqa
    data = prompt
    user_message += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data
        user_message += data
    else:
        pass
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )

    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


def apply_apps_map(dataset: Dataset) -> Dataset:
    apps_conversations = dataset.map(map_apps_conversations)
    return apps_conversations


def map_taco_conversations(x):
    """Map the TACO dataset to the required format."""
    test_case = json.loads(x["input_output_x"])
    starter_code = x["starter_code"]
    prompt = x["question"]

    user_message = ""
    data = test_case
    if not data.get("fn_name"):
        user_message += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # "\nUse Standard Input format"#\n" #noqa
    else:
        user_message += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # "\nUse Call-Based format"#\n" #noqa
    data = prompt
    user_message += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data
        user_message += data
    else:
        pass
    assistant_message = (
        f"<|begin_of_thought|>\n\n{x['reasoning']}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{x['deepseek_solution']}\n\n<|end_of_solution|>"
    )

    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": user_message},
            {"from": "assistant", "value": assistant_message},
        ],
    }


def apply_taco_map(dataset: Dataset) -> Dataset:
    taco_conversations = dataset.map(map_taco_conversations)
    return taco_conversations


def map_still2_conversations(x):
    """Map the still2 dataset to the required format."""
    return {
        "system": SKY_T1_SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": x["question"]},
            {"from": "assistant", "value": x["combined_text"]},
        ],
    }


def apply_still2_map(dataset: Dataset) -> Dataset:
    still2_conversations = dataset.filter(lambda x: x["domain"] in ["puzzle", "physics", "biology", "chemistry"]).map(map_still2_conversations)
    return still2_conversations
