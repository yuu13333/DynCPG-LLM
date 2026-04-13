import os
from typing import Literal, List
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState, START  # MessagesState是LangGraph内置的消息状态类
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from typing import Sequence
from typing_extensions import Annotated
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from openai import OpenAI
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
from joern import JoernSession, create_cpg
from logger import setup_thread_logger
from download_repository import download_commit_snapshot
from node_reviewer import reviewer_node
from node_critic import critic_node
from node_retriever import retriever_node

base_dir = "./result"
os.environ['TERM'] = 'dumb'

class Dialog(BaseModel):
    role: str = Field(description="The role of the model (either 'developer' or 'reviewer').")
    content: str = Field(description="The content of the response from the role.")


# Graph state
class State(TypedDict):
    role: str  # reviewer or developer, retriever need this info to router
    flag: bool  # vulnerable or non-vulnerable
    idx: str  # sample idx
    func_code: str  # function code
    CR_dialog: list[Dialog]  # dialogs between reviewer and developer
    messages: list[Dialog]  # reviewer messages
    final_result: list[any]  # final detection result of security defects
    abandoned_result: list[any]
    context: str
    j_server: any
    router: any


def reviewer_router(state: State) -> State:
    """
    流程路由（类似交通交警）：根据LLM的输出，决定下一个节点
    :param state: 流程状态
    :return: 下一个节点名称（"tools" 或 END）
    """
    thread_name = threading.current_thread().name  # 获取当前线程的名字
    print(f"thread {thread_name} - [Run reviewer_router]")
    messages = state["messages"]
    action = json.loads(messages[-1].content)
    if "missing_context" in action and len(action["missing_context"]) > 0:
        state["context"] = action["missing_context"]
        state["router"] = "retriever_node"
    else:
        idx = state["idx"]
        flag = state["flag"]
        dialogs = state["CR_dialog"]
        dialogs.append(Dialog(role="reviewer", content=messages[-1].content))
        state["CR_dialog"] = dialogs
        state["context"] = ""

        thinking_file = base_dir + "/" + idx + "/" + ("vulnerable" if flag else "non-vulnerable") + "_reviewer.jsonl"
        reviewer_thought = []
        messages = state["messages"]
        for msg in messages:
            reviewer_thought.append({"role": msg.role, "content": msg.content})
        stored_info = {"round": str((len(dialogs) + 1) // 2), "reviewer_thought": reviewer_thought}
        with open(thinking_file, "a", encoding='utf-8') as f1:
            f1.write(json.dumps(stored_info, ensure_ascii=False) + '\n')
        if len(action["vulnerability"]) == 0:
            state["router"] = END
        else:
            state["router"] = "critic_node"
    return state


def retriever_router(state: State) -> State:
    thread_name = threading.current_thread().name  # 获取当前线程的名字
    print(f"thread {thread_name} - [Run retriever_router]")
    if state["role"] == "Reviewer":
        state["router"] = "reviewer_node"
    else:
        state["router"] = END
    return state


def critic_router(state: State) -> State:
    dialogs = state["CR_dialog"]
    if len(dialogs) >= 6:
        # 超过三轮次
        state["router"] = END
    else:
        state["messages"] = []
        state["router"] = "reviewer_node"
    return state


workflow = StateGraph(State)
workflow.add_node("reviewer_node", reviewer_node)
workflow.add_node("retriever_node", retriever_node)
workflow.add_node("reviewer_router", reviewer_router)
workflow.add_node("retriever_router", retriever_router)
workflow.add_node("critic_node", critic_node)
workflow.add_node("critic_router", critic_router)
workflow.add_edge(START, "reviewer_node")
# 条件流转：从chat_bot节点出发，由tool_router决定下一步
workflow.add_edge("reviewer_node", "reviewer_router")
workflow.add_conditional_edges("reviewer_router", lambda state: state["router"],
                               {END: END, "retriever_node": "retriever_node",
                                "reviewer_node": "reviewer_node", "critic_node": "critic_node"})
workflow.add_edge("retriever_node", "retriever_router")
workflow.add_conditional_edges("retriever_router", lambda state: state["router"],
                               {END: END, "retriever_node": "retriever_node",
                                "reviewer_node": "reviewer_node"})
workflow.add_edge("critic_node", "critic_router")
workflow.add_conditional_edges("critic_router", lambda state: state["router"],
                               {END: END, "retriever_node": "retriever_node",
                                "reviewer_node": "reviewer_node"})


def non_stream_graph_updates(sample: dict, flag: bool):
    thread_name = threading.current_thread().name  # 获取当前线程名
    logger = setup_thread_logger(thread_name)  # 为当前线程设置日志

    idx = str(sample["idx"])
    res_dir = base_dir + "/" + idx
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    project_url = sample["project_url"]
    if flag:
        commit_id = [sample["vulnerability_introducing_commit_id"]]
        func_code = sample["vulnerable_function_body"]
    else:
        commit_id = [sample["vulnerability_fixing_commit_id"]]
        func_code = sample["non_vulnerable_function_body"]

    owner_repo = project_url[project_url.find("github.com/") + len("github.com/"):]
    owner, repo = owner_repo.split("/")
    save_dir = "./project"
    joern = None
    bin_url = None
    try:
        # 获取仓库代码
        project_dir = repo + "-" + commit_id[0]
        project_url = save_dir + "/" + project_dir
        k = download_commit_snapshot(logger, owner, repo, commit_id[0], save_dir)
        if k:
            print(f"thread {thread_name} -【download】 Project {project_url}.")
        else:
            print(f"thread {thread_name} -【failed to download】 Project {project_url}.")
            return

        # 运行joern
        joern = JoernSession(logger)
        # 生成cpg图
        create_cpg(logger, joern, project_url, project_dir)
        bin_url = "./workspace/" + project_dir
        logger.info(f"【Create CPG graph of {bin_url}.")
        print(f"thread {thread_name} -【Create】 CPG graph of {bin_url}.")
        retry = 0
        while retry < 3:
            retry += 1
            try:
                # 同步阻塞，等待完整结果
                result = graph.invoke({
                    "role": "Reviewer",
                    "flag": flag,
                    "idx": idx,
                    "func_code": func_code,
                    "CR_dialog": [],
                    "messages": [],
                    "final_result": [],
                    "abandoned_result": [],
                    "context": "",
                    "j_server": joern,
                    "router": START
                })

                # 存储CR_dialog
                CR_dialogs = result["CR_dialog"]
                CR_dialog = []

                for conv in CR_dialogs:
                    CR_dialog.append({"role": conv.role, "content": conv.content})

                # 存储final_result
                stored_info = {"final_result": {"vulnerability": result["final_result"]}, "CR_dialog": CR_dialog}
                file_path = res_dir + "/" + ("vulnerable" if flag else "non-vulnerable") + ".json"

                with open(file_path, "w") as f1:
                    json.dump(stored_info, f1, indent=4)
                print("Assistant:", result["final_result"])
                break
            except Exception as e:
                print(e)

    except Exception as e:
        print(e)

    # 删除cpg图
    if bin_url and os.path.exists(bin_url):
        shutil.rmtree(bin_url)
        logger.info(f"【Delete】CPG graph of {bin_url}.")
        print(f"thread {thread_name} -【Delete】CPG graph of {bin_url}.")

    # 删除代码仓库
    if os.path.exists(project_url):
        shutil.rmtree(project_url)
        logger.info(f"【Delete】Project {project_url}.")
        print(f"thread {thread_name} -【Delete】Project {project_url}.")

    # 停止joern进程
    if joern:
        joern.close()
    return


def process_single_item(item):
    """处理单个样本：调用两次 non_stream_graph_updates"""
    idx = str(item["idx"])
    try:
        if not os.path.exists(f"./result/{idx}/vulnerable.json"):
            non_stream_graph_updates(item, True)
            print("[vulnerable done]")
        else:
            print("[pass vulnerable]")
        if not os.path.exists(f"./result/{idx}/non-vulnerable.json"):
            non_stream_graph_updates(item, False)
            print("[non-vulnerable done]")
        else:
            print("[pass non-vulnerable]")
        return True
    except Exception as e:
        print(f"[Thread {threading.get_ident()}] Error processing item: {e}")
        return False


def main():
    items = []
    with open("evaluated_benchmark.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    random.shuffle(items)
    total = len(items)
    print(f"✅ Loaded {total} items. Starting multithreaded processing...")

    max_workers = min(5, (os.cpu_count() or 10))
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_single_item, item): i
            for i, item in enumerate(items)
        }

        # 可选：使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_item), total=total, desc="Processing"):
            idx = future_to_item[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
            except Exception as exc:
                print(f"Item {idx} generated an exception: {exc}")

    print(f"\n🎉 All done! Successfully processed {success_count}/{total} items.")


if __name__ == "__main__":
    # 编译图（在主线程中执行一次）
    graph = workflow.compile()

    # 执行多线程处理
    main()
