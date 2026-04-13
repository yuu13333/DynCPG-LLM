import json
import shutil
import os
import threading

from training_set_prepration.download_repository import download_commit_snapshot
from training_set_prepration.joern import JoernSession, create_cpg
from training_set_prepration.logger import setup_thread_logger
from training_set_prepration.workflow_steps import step2_generate_cpg_queries, step4_validate_context, \
    step1_generate_context_desc

os.environ['TERM'] = 'dumb'
DONE = []


def get_context_info_agent(joern, context_desc, idx_info, sample_res_path, logger, fun_code):
    thread_name = threading.current_thread().name  # 获取当前线程
    logger.info(f"Target context:{context_desc}")
    print(f"thread {thread_name} - Target context:{context_desc}")
    context_info_res = None  # 最终的context_info结果
    cpg_queries_res = None  # 最终的cpg_queries结果
    validation_history = None
    error_history = None
    step2_res = None

    n = 0
    while n < 5:
        n += 1
        logger.info(f"--------Round {n} start-----------")
        step2_res = step2_generate_cpg_queries(logger, fun_code, context_desc, step2_res, error_history,
                                               validation_history)
        validation_history = None
        error_history = None
        if step2_res is not None:  # queries生成成功
            logger.info(f"STEP 2 - generate_cpg_queries: {step2_res}")

            cpg_out = []
            round_error_history = []  # 只存储上一轮次的出错语句
            is_error = False  # 标识是否出错的flag
            for query in step2_res:
                logger.info(f"RUN CPG QUERY:{query}")
                try:
                    cur_out = joern.run_block(query)
                    logger.info(f"QUERY RESULT:{cur_out}")
                    if "error" in cur_out or "Error" in cur_out:  # 出错
                        round_error_history.append({"query": query, "error_message": cur_out})
                        is_error = True
                    else:
                        cpg_out.append({"query": query, "result": cur_out})
                except Exception as e:
                    print(e)
                    print(f"thread {thread_name} - 【Failed】 to run the CPG query in joern")

            if is_error:
                error_history = round_error_history
                continue
            else:
                validation_res = step4_validate_context(logger, cpg_out, context_desc)
                logger.info(f"STEP 4 - validate_context: {validation_res}")
                print(f"thread {thread_name} - STEP 4 - validate_context: {validation_res}")
                if validation_res is not None:
                    context_info_res = cpg_out
                    cpg_queries_res = [item["query"] for item in context_info_res]
                    if validation_res["context_match"]:
                        break
                    else:
                        validation_history = {
                            "results": cpg_out,
                            "explanation": validation_res["explanation"]
                        }
                        continue
                else:
                    continue
        else:
            break

    sample = {"idx": idx_info, "context_desc": context_desc, "queries": cpg_queries_res,
              "context_info": context_info_res,
              "validation": "" if validation_history is None else validation_history["explanation"]}
    with open(sample_res_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return context_info_res


def run(path, sample_res_path):
    thread_name = threading.current_thread().name  # 获取当前线程的名字
    logger = setup_thread_logger(thread_name)  # 为当前线程设置日志
    save_dir = "./project"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
                item = json.loads(line)
                for i in item:
                    if i["idx"] in DONE:
                        print(f"thread {thread_name} - continue")
                        continue
                    commit_id = i["parent_hashes"]
                    project_url = i["project_url"]
                    owner_repo = project_url[project_url.find("github.com/") + len("github.com/"):]
                    owner, repo = owner_repo.split("/")
                    cwe = i["cwe"]
                    cve_desc = i["cve_desc"]
                    func_code = i["func"]
                    idx_info = i["idx"]
                    joern_running = False
                    print(f"thread {thread_name} - 【run {idx_info}】!")

                    # step1 reviewer推理
                    step1_res = step1_generate_context_desc(logger, func_code, cwe, cve_desc)
                    if "missing_context" in step1_res:
                        try:
                            # 获取仓库代码
                            project_dir = repo + "-" + commit_id[0]
                            project_url = save_dir + "/" + project_dir
                            k = download_commit_snapshot(logger, owner, repo, commit_id[0], save_dir)
                            if k:
                                print(f"thread {thread_name} -【download】 Project {project_url}.")
                            else:
                                print(f"thread {thread_name} -【failed to download】 Project {project_url}.")
                                continue

                            # 运行joern server
                            joern = JoernSession(logger)
                            joern_running = True
                            create_cpg(logger, joern, project_url, project_dir)
                            bin_url = "./workspace/" + project_dir
                            logger.info(f"【Create CPG graph of {bin_url}.")
                            print(f"thread {thread_name} -【Create】 CPG graph of {bin_url}.")

                            # 获取上下文
                            context_descs = step1_res["missing_context"]
                            logger.info("【Start to retrieve context information】")
                            print("thread {thread_name} - 【Start to retrieve context information】")
                            logger.info(f"STEP 1 - missing_context: {context_descs}")
                            print(f"thread {thread_name} - STEP 1 - missing_context: {context_descs}")
                            for context_desc in context_descs:
                                get_context_info_agent(joern, context_desc, idx_info, sample_res_path, logger, func_code)  # 传递logger是为了保证日志文件一致

                            # 删除cpg图
                            if os.path.exists(bin_url):
                                shutil.rmtree(bin_url)
                                logger.info(f"【Delete】CPG graph of {bin_url}.")
                                print(f"thread {thread_name} -【Delete】CPG graph of {bin_url}.")
                        except Exception as e:
                            print(f"thread {thread_name} -【Failed】 to generate the CPG graph")
                            print(e)

                        # 删除代码仓库
                        if os.path.exists(project_url):
                            shutil.rmtree(project_url)
                            logger.info(f"【Delete】Project {project_url}.")
                            print(f"thread {thread_name} -【Delete】Project {project_url}.")

                        # 停止joern进程
                        if joern_running:
                            joern.close()
                    else:
                        continue
                break

# 定义任务参数
tasks = []
max_thread_num = 7
for i in range(1,max_thread_num+1):
    tasks.append((f"thread_{i}.jsonl", f"training_set_{i}.jsonl"))

# 创建并启动线程
threads = []
for input_file, output_file in tasks:
    thread = threading.Thread(target=run, args=(input_file, output_file, -1))
    thread.start()
    threads.append(thread)

# 等待所有线程完成
for thread in threads:
    thread.join()