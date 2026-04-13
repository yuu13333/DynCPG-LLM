import re
import json
import threading
from get_llm_response import get_ds_response, get_query_response
from run import State, Dialog
import time
from postprocess import PATH_STRING, FIND_METHOD


def validate_query_result(context_desc, context_information):
    validate_prompt = f"""To query the code context information, a cpgql query has been executed using the Joern tool, and the response has been captured. Your task is to determine whether the cpgql query was successfully executed and whether the response is empty.

context_description:{context_desc}

query_response:{context_information}

OUTPUT FORMAT: strictly in JSON
If any query encounters an error or exception, set 'query_status' to 'failure', else 'success'
If any query obtains empty response (e.g., val res1: List[String] = List()), set 'is_empty' to true, else false.  
{{
    "query_status": <string, "failure" or "success">, 
    "is_empty": <boolean, true or false>,
    "explanation": "a brief reasoning of your judgement."}}
"""

    llm_messages = [
        {"role": "user", "content": validate_prompt},
    ]
    response = get_ds_response(llm_messages, 0)
    return json.loads(response.choices[0].message.content)


def retriever_node(state: State) -> State:
    thread_name = threading.current_thread().name  # 获取当前线程的名字
    print(f"thread {thread_name}------------------【run context_retriever】------------------")
    messages = state["messages"]
    joern_server = state["j_server"]
    context_desc = state["context"]
    func_code = state["func_code"]

    retrieved_information = ""
    times = 0
    while times < 6:
        times += 1
        # 调用model
        cpg_query_prompt = f"""Given a function's code and the corresponding CPG graph generated from the complete code repository using the Joern tool, please create one or more CPG queries to retrieve the required context information as described below.
    Constraints:
    - Queries must be executable in Joern/CPGQL.
    - Use Scala language features to construct the queries.
    - The number of queries should be minimized, avoiding duplicates or similar queries.

    Output Format:
    Please output the queries as an array, like:
    ["Query 1", "Query 2"]
    Given code:{func_code}
    Required Context Description: {context_desc}"""

        queries = get_query_response(cpg_query_prompt)

        context_information = []

        # 运行得到内容
        def replace_l_with_dedup(query):
            # 使用正则表达式判断字符串是否以 .l 结尾，并替换为 .dedup.l
            return re.sub(r'\.l$', '.dedup.l', query)

        for query in queries:
            query = replace_l_with_dedup(query)
            try:
                # 处理paths
                pattern = r".*?val\s+paths\s*=\s*.*\.reachableByFlows\((.*)\)"
                print(f"thread {thread_name} - {query}")
                # 检查输入字符串是否匹配正则表达式
                if re.match(pattern, query):
                    joern_server.run_block(query)
                    joern_server.run_block(FIND_METHOD)
                    cur_out = joern_server.run_block(PATH_STRING)  # 会出现\r\n去不掉的情况
                else:
                    cur_out = joern_server.run_block(query)
                context_information.append({"query": query, "result": cur_out})
            except Exception as e:
                print(e)
                print(f"thread {thread_name} - 【Failed】 to run the CPG query in joern")
                context_information.append({"query": query, "result": "【Failed】 to run the CPG query in joern"})
                time.sleep(1)

        final_context_information = ""
        for idx, context_info in enumerate(context_information):
            q, r = context_info["query"], context_info["result"]
            final_context_information += f"Query{str(idx)}: {q}\nResult{str(idx)}: {r}\n"
        flag = validate_query_result(context_desc, final_context_information)
        print(final_context_information)
        print(flag)
        if flag["query_status"] == "success":
            retrieved_information = final_context_information
            if times == 6 or flag["is_empty"] == False:
                break

    if len(retrieved_information) == 0:
        retrieved_information = "Failed to retrieve the context. Please change the context you request."
    messages.append(Dialog(role="Context_retriever", content=retrieved_information))
    print(f"thread {thread_name} - get final_context_information: {retrieved_information}")
    state["messages"] = messages
    state["context"] = ""
    return state
