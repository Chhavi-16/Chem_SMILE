import subprocess
import sys

print("Starting Pinggy tunnel...")
cmd = "ssh -o StrictHostKeyChecking=no -p 443 -R 0:localhost:8501 free@a.pinggy.io"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

for line in iter(p.stdout.readline, ''):
    print(line, end='', flush=True)
    if "https://" in line:
        with open("pinggy_link.txt", "w", encoding="utf-8") as f:
            f.write(line.strip())
