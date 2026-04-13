import os
import pexpect
import re

os.environ['TERM'] = 'dumb'


class JoernSession:
    def __init__(self, logger, joern_cmd="joern"):
        self.child = pexpect.spawn(
            joern_cmd,
            encoding="utf-8",
            timeout=1800
        )
        # 等 joern 提示符
        self.child.expect("joern>")
        self.logger = logger
        self.logger.info("Joern shell started")

    def run_block(self, block: str):
        """
        发送一整个 CPGQL block（多行）
        """
        self.child.sendline(block)
        self.child.expect("joern>")
        output = self.child.before.strip()
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)  # 清除颜色控制字符
        clean_output = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', clean_output)  # 清除其他控制字符

        echoed_query = block.replace('\n', '\r\n')
        if clean_output.startswith(echoed_query):
            print("clean here")
            result = clean_output[len(echoed_query):].strip()
            return result
        else:
            print(f"clean_output:{clean_output}\nechoed_query:{echoed_query}")
            parts = clean_output.split('\r\n', 1)  # 第二个参数 1 表示只分割一次
            if len(parts) > 1:
                return parts[1]  # 返回第一个 '\r\n' 之后的内容 查询后的内容
            else:
                return ""  # 如果没有找到 '\r\n'，返回空字符串
            # return clean_output

    def close(self):
        self.child.sendline(":quit")
        self.child.close()
        self.logger.info("Joern shell closed")


def create_cpg(logger, joern, project_path, project_name):
    print("create cpg graph!")
    create_cmd = f'importCode("{project_path}","{project_name}")'
    out = joern.run_block(create_cmd)
    logger.info(out)
