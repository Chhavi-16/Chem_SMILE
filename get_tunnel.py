import subprocess
import sys

print("Starting localtunnel...")
p = subprocess.Popen("lt --port 8501", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

for line in iter(p.stdout.readline, ''):
    print(line, end='', flush=True)
    if "url" in line.lower() or "https://" in line:
        with open("live_link.txt", "w") as f:
            f.write(line.strip())
        break
