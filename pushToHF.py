from huggingface_hub import HfApi, HfFolder, Repository
from transformers import AutoTokenizer

# Pfad zu Ihrem Modellverzeichnis
model_directory = "./HuggingfacePush"

# Ihr Hugging Face Benutzername und der Name des Modells
hf_username = "KennyDain"
model_name = "Llama3_MedQA_bnb_4bit"

# Vollständiger Modellname mit Benutzername
full_model_name = f"{hf_username}/{model_name}"

# HfApi Instanz erstellen
api = HfApi()

# Repository auf Hugging Face erstellen
api.create_repo(repo_id=full_model_name, exist_ok=True)

# Klonen des Repositories in das lokale Verzeichnis
repo = Repository(local_dir=model_directory, clone_from=full_model_name)

# Alle Dateien im Verzeichnis in das Repository pushen
repo.push_to_hub(commit_message="Add model files")

# Laden und Pushen des Tokenizers (Annahme, dass PreTrainedTokenizerFast verwendet wird)
tokenizer = AutoTokenizer.from_pretrained(model_directory)
tokenizer.push_to_hub(full_model_name)
