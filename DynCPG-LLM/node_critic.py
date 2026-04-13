import json
import threading
from prompt_critic import CRITIC_SYSTEM_PROMPT, CRITIC_VUL_CASE, CRITIC_OUTPUT_FORMAT_PROMPT_VUL_CASE
from get_llm_response import get_ds_response
from run import State, Dialog


def critic_node(state: State) -> State:
    thread_name = threading.current_thread().name  # 获取当前线程
    print(f"thread {thread_name}------------------【run Critic】------------------")
    function_code = state["func_code"]
    dialogs = state["CR_dialog"]
    messages = state["messages"]

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

    reviewer_res = json.loads(dialogs[-1].content)["vulnerability"]
    reviewer_findings = {"is_vulnerable": True, "vulnerability": reviewer_res}
    critic_prompt = f"""{CRITIC_SYSTEM_PROMPT}

AVAILABLE INFORMATION：
Given function code: 
{function_code}

{CRITIC_VUL_CASE}

Reviewer’s reasoning and findings: 
{history_contexts}

{CRITIC_OUTPUT_FORMAT_PROMPT_VUL_CASE}"""
    llm_messages = [
        {"role": "user", "content": critic_prompt},
    ]
    # print(critic_prompt)
    response = get_ds_response(llm_messages, 0)
    res = response.choices[0].message.content
    print(f"thread {thread_name} - [Critic Response:]\n {res}\n")
    dialogs.append(Dialog(role="Critic", content=res))
    state["CR_dialog"] = dialogs
    # vul
    vul_items = json.loads(res)["vulnerability"]
    add_final_result, add_abandoned_result = [], []
    for item in vul_items:
        if item["judgement"] == "ACCEPT":
            new_item = {
                "vulnerability_type": item["vulnerability_type"],
                "cwe_id": item["cwe_id"],
                "location": item["location"],
                "explanation": item["explanation"]
            }
            add_final_result.append(new_item)
        else:
            add_abandoned_result.append(item)

    tmp = state["final_result"]
    tmp.extend(add_final_result)
    state["final_result"] = tmp  # 要求final_result是list

    tmp2 = state["abandoned_result"]
    tmp2.extend(add_abandoned_result)
    state["abandoned_result"] = tmp2
    return state