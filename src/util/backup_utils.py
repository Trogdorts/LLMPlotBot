
"""
Backup utilities: create timestamped backups of the current working directory.
"""

import os
import shutil
from datetime import datetime
from typing import List


def create_backup(backup_dir: str, ignore_list: List[str], logger):
    """
    Create a timestamped backup of the current working directory, excluding folders in ignore_list.
    """
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(backup_dir, f"backup_{ts}")
    os.makedirs(dest, exist_ok=True)
    root = os.getcwd()
    logger.info(f"Creating backup at {dest}")

    for item in os.listdir(root):
        if any(ig.lower() in item.lower() for ig in ignore_list):
            logger.debug(f"Skipping backup for ignored item: {item}")
            continue
        src_path = os.path.join(root, item)
        dst_path = os.path.join(dest, item)
        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        except Exception as e:
            logger.warning(f"Backup skip {item}: {e}")
    logger.info(f"Backup complete: {dest}")
