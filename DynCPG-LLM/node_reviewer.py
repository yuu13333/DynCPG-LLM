import threading
from prompt_reviewer import SYSTEM_PROMPT, ANALYSIS_RULES_PROMPT, OUTPUT_FORMAT_PROMPT_CRITIC, OUTPUT_FORMAT_PROMPT
from get_llm_response import get_ds_response
from run import State, Dialog


def reviewer_node(state: State) -> State:
    thread_name = threading.current_thread().name  # 获取当前线程
    print(f"thread {thread_name}------------------【run Reviewer】------------------")
    function_code = state["func_code"]
    messages = state["messages"]
    final_result = state["final_result"]
    abandoned_result = state["abandoned_result"]

    history_contexts = ""
    step = 1
    for msg in messages:
        if len(history_contexts) > 0:
            history_contexts += "\n"
        if msg.role == "Reviewer":
            history_contexts += f"Step {step}: {msg.content}"
            step += 1
        else:
            history_contexts += f"Retrieved context: {msg.content}"

    constraint = f"""\nCONSTRAINT:
Previously Identified Vulnerabilities: {final_result}
Previously Rejected Findings: {abandoned_result}
You must focus on identifying NEW security defects beyond those listed above. Avoid reporting duplicate or similar issues.\n"""

    reviewer_prompt = f"""{SYSTEM_PROMPT}

AVAILABLE INFORMATION：
Given function code: {function_code}
{constraint if len(final_result) > 0 or len(abandoned_result) > 0 else ""}
{ANALYSIS_RULES_PROMPT}

{OUTPUT_FORMAT_PROMPT_CRITIC if len(final_result) > 0 or len(abandoned_result) > 0 else OUTPUT_FORMAT_PROMPT}

{history_contexts}
Next Step:"""
    # print(reviewer_prompt)
    llm_messages = [
        {"role": "user", "content": reviewer_prompt},
    ]
    response = get_ds_response(llm_messages, 0)
    res = response.choices[0].message.content
    print(f"thread {thread_name} - [Reviewer Response:]\n {res}\n")
    messages.append(Dialog(role="Reviewer", content=res))
    state["messages"] = messages
    return state