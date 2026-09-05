import subprocess
import time

def start_tunnel():
    print("Launching persistent tunnel...")
    p = subprocess.Popen("lt --port 8501 --subdomain smile-chem-intel", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(p.stdout.readline, ''):
        print(line, end='', flush=True)

if __name__ == "__main__":
    start_tunnel()
