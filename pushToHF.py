from huggingface_hub import HfApi, HfFolder, Repository
from transformers import PreTrainedTokenizerFast

# Pfad zu Ihrem Modellverzeichnis
model_directory = "."

# Ihr Hugging Face Benutzername und der Name des Modells
hf_username = "KennyDain"
model_name = "Llama3_8B_bnb_4bit_MedQA"

# Erstellen des Repository-Objekts
repo_url = HfApi().create_repo(name=model_name, token=HfFolder.get_token(), exist_ok=True)
repo = Repository(local_dir=model_directory, clone_from=repo_url)

# Push der Dateien zum Hub
repo.push_to_hub(commit_message="Add model files")

# Laden und Pushen des Tokenizers (Annahme, dass PreTrainedTokenizerFast verwendet wird)
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
tokenizer.push_to_hub(model_name)
