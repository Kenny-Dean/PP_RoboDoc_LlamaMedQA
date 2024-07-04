import json
import torch
import socket
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Model and save directory
model_name = "KennyDain/Llama3_MedQA_bnb_4bit"
save_directory = "./model"

# Global variable to track if model is loaded
model_loaded = False
model = None
tokenizer = None

def download_and_save_model(model_name, save_directory):
    try:
        # Load model and tokenizer from Hugging Face
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Save tokenizer and model to disk
        tokenizer.save_pretrained(save_directory)
        model.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory}")
        return True

    except Exception as e:
        print(f"Failed to download and save model: {str(e)}")
        return False

def load_model():
    global model, tokenizer, model_loaded
    if not model_loaded:
        print("Loading model into VRAM...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=save_directory,
                max_seq_length=2048,
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")

def unload_model():
    global model, model_loaded
    if model_loaded:
        del model
        torch.cuda.empty_cache()
        model_loaded = False
        print("Model unloaded from VRAM.")

def chat_with_robodoc(user_input, chat_history, nodes_from_subgraph=None, image_captioning=None):
    global model_loaded, tokenizer
    system_message = [
        {
            "role": "system",
            "content": "You are an AI assistant supporting a doctor. The user describes patient symptoms. You receive node names and node types from a knowledge graph for further reference. These are not confirmed diseases or drugs. Limit responses to 200 characters. For suspected diseases, ask for specific details."
        }
    ]

    if nodes_from_subgraph:
        system_message[0]["content"] += f"\n\nNode Info:\n{nodes_from_subgraph}"

    if image_captioning:
        system_message[0]["content"] += f"\n\nThe user has also uploaded a picture to better describe his symptoms. Described picture:\n{image_captioning}"

    # Construct prompt from updated chat history
    content_str = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in chat_history])
    messages = [
        {"role": "system", "content": system_message[0]["content"] + "\n" + content_str},
        {"role": "user", "content": user_input}
    ]

    try:
        load_model()

        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        prompt_text = alpaca_prompt.format(
            "Continue the dialogue with the patient.",
            "\n".join([f"{message['role']}: {message['content']}" for message in chat_history]),
            user_input
        )

        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        model_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        chat_history.append({"role": "user", "content": user_input})

    except Exception as e:
        print(f"Exception occurred during model generation: {str(e)}")
        model_response = "Sorry, I couldn't process your request at the moment."

    finally:
        unload_model()

    # Append model response to chat history
    chat_history.append({"role": "assistant", "content": model_response})

    return user_input, model_response, chat_history

def listen_for_prompts():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 65143))
        s.listen()
        print("Waiting for connections...")
        while True:  # Keep listening for connections indefinitely
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:  # Keep processing prompts within the connection
                    data = conn.recv(4096)
                    if not data:
                        break
                    received_data = json.loads(data.decode('utf-8'))
                    user_input = received_data['user_input']
                    chat_history = received_data['chat_history']
                    nodes_from_subgraph = received_data.get('nodes_from_subgraph', None)
                    image_captioning = received_data.get('image_captioning', None)
                    user_question, model_response, updated_history = chat_with_robodoc(user_input, chat_history, nodes_from_subgraph, image_captioning)
                    response_data = {
                        "user_input": user_question,
                        "model_response": model_response,
                        "chat_history": updated_history
                    }
                    conn.sendall(json.dumps(response_data).encode('utf-8'))

if __name__ == "__main__":
    # Download and save the model if not already saved
    if not download_and_save_model(model_name, save_directory):
        print("Failed to download and save model. Exiting...")
    else:
        # Start listening for prompts
        listen_for_prompts()
