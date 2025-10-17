"""Backup utilities: create timestamped backups of the current working directory."""


import os
import zipfile
from datetime import datetime
from typing import List

from .path_utils import normalize_for_logging


def create_backup(backup_dir: str, ignore_list: List[str], logger):
    """
    Create a timestamped backup of the current working directory, excluding folders in ignore_list.
    """
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest_base = os.path.join(backup_dir, f"backup_{ts}")
    dest = f"{dest_base}.zip"
    root = os.getcwd()
    logger.info("Creating backup at %s", normalize_for_logging(dest, extra_roots=[root]))

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in os.listdir(root):
            if any(ig.lower() in item.lower() for ig in ignore_list):
                logger.debug(f"Skipping backup for ignored item: {item}")
                continue

            src_path = os.path.join(root, item)

            try:
                if os.path.isdir(src_path):
                    for dirpath, dirnames, filenames in os.walk(src_path):
                        dirnames[:] = [
                            d
                            for d in dirnames
                            if not any(
                                ig.lower() in os.path.join(dirpath, d).lower()
                                for ig in ignore_list
                            )
                        ]

                        # Preserve the directory structure relative to the project root.
                        rel_dirpath = os.path.relpath(dirpath, root)
                        if not filenames and not dirnames:
                            zf.writestr(f"{rel_dirpath}/", "")
                            continue

                        for filename in filenames:
                            file_path = os.path.join(dirpath, filename)
                            if any(ig.lower() in file_path.lower() for ig in ignore_list):
                                logger.debug(
                                    "Skipping backup for ignored item: %s",
                                    os.path.relpath(file_path, root),
                                )
                                continue

                            arcname = os.path.relpath(file_path, root)
                            zf.write(file_path, arcname)
                else:
                    zf.write(src_path, item)
            except Exception as e:
                logger.warning("Backup skip %s: %s", item, e)

    logger.info("Backup complete: %s", normalize_for_logging(dest, extra_roots=[root]))
