from pathlib import Path
import os
import runpy

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

runpy.run_path(
    str(BASE_DIR / "src" / "05_streamlit_dashboard.py"),
    run_name="__main__"
)
