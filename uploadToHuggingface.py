from huggingface_hub import HfApi, HfFolder, Repository

# Definiere den Pfad zu deinem Modellverzeichnis
model_dir = "./Llama3_8B_bnb_4bit_RoboDoc_MedQA"

# Name des Repositorys
repo_name = "KennyDain/Llama3_8B_bnb_RoboDoc_MedQA"

# Initialize a new Repository instance
repo = Repository(local_dir=model_dir, clone_from=repo_name)

# Add all files in the model directory to the repository
repo.git_add()

# Commit the files
repo.git_commit("Upload fine-tuned model")

# Push the files to the Hugging Face Model Hub
repo.git_push()
