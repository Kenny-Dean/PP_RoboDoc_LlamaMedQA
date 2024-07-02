import subprocess
import os

# Definiere den Pfad zu deinem Modellverzeichnis und das Repository-URL
model_dir = "/home/pp/RoboDoc/PP_RoboDoc_LlamaMedQA/Llama3_8B_bnb_4bit_RoboDoc_MedQA"
repo_url = "https://huggingface.co/KennyDain/Llama3_8B_bnb_RoboDoc_MedQA"

# Wechsle in das Modellverzeichnis
os.chdir(model_dir)

# Überprüfen, ob der Remote-Name 'origin' existiert
result = subprocess.run(["git", "remote"], capture_output=True, text=True)
if 'origin' in result.stdout:
    subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True)
else:
    subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)

# Git LFS installieren und tracken
subprocess.run(["git", "lfs", "install"], check=True)
subprocess.run(["git", "lfs", "track", "adapter_model.safetensors"], check=True)

# Füge alle Dateien hinzu
subprocess.run(["git", "add", "."], check=True)

# Commite die Dateien
subprocess.run(["git", "commit", "-m", "Upload fine-tuned model"], check=True)

# Pushe die Dateien zum Hugging Face Model Hub
subprocess.run(["git", "push", "origin", "master"], check=True)
