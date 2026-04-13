import zipfile
import os
import subprocess


def unzip_commit_snapshot(logger, zip_path, save_dir):
    # 解压到指定目录
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
        logger.info(f"Unzipped: {zip_path} to {save_dir}")


def download_commit_snapshot(logger, owner, repo, commit_hash, save_dir):
    url = f"https://github.com/{owner}/{repo}/archive/{commit_hash}.zip"
    # 构建 curl 命令
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 保存位置
    zip_path = os.path.join(save_dir, f"{owner}_{repo}_{commit_hash}.zip")
    curl_command = ['curl', '-L', '-s', url, '-o', zip_path]
    # 执行 curl 命令
    try:
        subprocess.run(curl_command, check=True)
        # 解压缩 ZIP 文件并删除
        unzip_commit_snapshot(logger, zip_path, save_dir)
        os.remove(zip_path)  # 删除 ZIP 文件
        logger.info(f"Deleted ZIP file: {zip_path}")
        logger.info(f"Downloaded: {owner}/{repo} - {commit_hash}")
        return True
    except Exception as e:
        logger.info(f"Failed to download: {owner}/{repo} - {commit_hash}, Exception: {e}")
    return False
