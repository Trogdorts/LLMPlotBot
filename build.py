
"""
Simple build script to create llm_batch_processor_build.zip from the project folder.
"""

import os
import zipfile

def build():
    base = "llm_batch_processor"
    zip_name = "llm_batch_processor_build.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(base):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, base))
    print(f"Created {zip_name}")

if __name__ == "__main__":
    build()
