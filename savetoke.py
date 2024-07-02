from transformers import AutoTokenizer

# Angenommen, du hast den Tokenizer aus dem Checkpoint geladen
checkpoint_path = "./outputs"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Definiere den Pfad, wo du den Tokenizer speichern möchtest
save_path = "./Llama3_8B_bnb_4bit_RoboDoc_MedQA_Tokenizer"

# Speichere den Tokenizer auf der Festplatte
tokenizer.save_pretrained(save_path)
