from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
model_name = "gpt2"  # You can choose a different GPT-3 variant

@app.route("/generate", methods=["POST"])
def generate_text():
    input_text = request.json.get("text", "")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=50256)

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
