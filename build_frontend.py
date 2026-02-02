import subprocess
import sys
from pathlib import Path

frontend_dir = Path("E:/Anima_Trainer/gui/frontend")

print("Building frontend...")
result = subprocess.run(
    ["npm", "run", "build"],
    cwd=str(frontend_dir),
    capture_output=True,
    shell=True,
    encoding="utf-8",
    errors="replace"
)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
