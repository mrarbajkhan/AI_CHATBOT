from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize Flask app
app = Flask(__name__)

# Route for chat
@app.route("/")
def index():
    return render_template('chat.html') 

# Chat route to handle messages
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response_text = get_Chat_response(msg)
    return jsonify(response_text)

# Function to handle chat response
def get_Chat_response(text):
    # Initial chat history is None
    chat_history_ids = None

    # Encode user input and add eos_token
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # If there's chat history, concatenate it
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate response with max token limit
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and return bot's response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
