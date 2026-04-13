import json
import os

from training_set_prepration.call_llm import get_response
from training_set_prepration.prompt import STEP1_SYSTEM_PROMPT, STEP2_SYSTEM_PROMPT, STEP4_SYSTEM_PROMPT


def step1_generate_context_desc(logger, func_code_tmp, cwe, cve_desc):
    system_prompt = STEP1_SYSTEM_PROMPT
    user_prompt = f"""\nFunction Code: {func_code_tmp}\n Vulnerability: {cwe} {cve_desc}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"STEP 1 - PROMPT: {messages}")
    step1_res = {}
    time = 0
    while time < 5:
        time += 1
        try:
            response = get_response(messages, 1.0)
            step1_res = json.loads(response.choices[0].message.content)
            break
        except Exception as e:
            print(f"Retrying context desc generation...{e}")
            continue
    return step1_res


def step2_generate_cpg_queries(logger, fun_code, context_desc_tmp, queries_tmp=None, error_history_tmp=None,
                               validation_history_tmp=None):
    system_prompt = STEP2_SYSTEM_PROMPT
    user_prompt = f"""Given code: {fun_code}\nRequired Context Description: {context_desc_tmp}"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if queries_tmp is not None:
        messages.append({"role": "assistant", "content": str({"queries": queries_tmp})})
        if error_history_tmp is not None:
            error_prompt = "The following error message was received during execution. Please fix your answer.\n"
            for error_out in error_history_tmp:
                query, error_msg = error_out["query"], error_out["error_message"]
                error_prompt += f"query:{query}\nerror_msg:{error_msg}"
            messages.append({"role": "user", "content": error_prompt})
        elif validation_history_tmp is not None:
            valid_prompt = f"""The query results do not match the required code context. Please fix your answer.
            Query Results: {validation_history_tmp["results"]}
            Explanation: {validation_history_tmp["explanation"]}"""
            messages.append({"role": "user", "content": valid_prompt})
        else:
            print("situation error!")
    logger.info(f"STEP 2 - PROMPT: {messages}")

    step2_res = None
    time = 0
    while time < 5:
        time += 1
        try:
            response = get_response(messages, 0.7)
            r = response.choices[0].message.content
            step2_res = json.loads(r)["queries"]
            break
        except Exception as e:
            print(f"Retrying queries generation...{e}")
            continue
    return step2_res


def step4_validate_context(logger, code_context_tmp, context_desc_tmp):
    system_prompt = STEP4_SYSTEM_PROMPT
    user_prompt = f"""Required Context: {context_desc_tmp}\nQuery Results:"""
    for cur_out in code_context_tmp:
        query, result = cur_out["query"], cur_out["result"]
        user_prompt += f"{result}\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"STEP 4 - PROMPT: {messages}")
    step4_res = None
    time = 0
    while time < 3:
        time += 1
        try:
            response = get_response(messages, 1.3)
            r = json.loads(response.choices[0].message.content)
            if "context_match" in r:
                step4_res = r
                break
            else:
                continue
        except Exception as e:
            print(f"valid error:{e}")
            continue
    return step4_res