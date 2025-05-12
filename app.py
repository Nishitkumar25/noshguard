
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
from mistralai import Mistral

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# API key for Mistral
api_key = os.getenv("MISTRAL_API_KEY", "aCHLMXr7v1Ojf2lniUNkjftHGnzxLoDp")
client = None

if api_key:
    try:
        client = Mistral(api_key=api_key)
        print("✅ Mistral client initialized")
    except Exception as e:
        print(f"❗ Error initializing Mistral client: {e}")
        client = None

@app.route("/")
def home():
    return jsonify({"message": "NoshGuard Python Backend is running!"})

@app.route("/ocr", methods=["POST"])
def run_ocr():
    if not client:
        return jsonify({"error": "Mistral client not initialized"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files["image"]
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        image_bytes = file.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        base64_url = f"data:{file.mimetype};base64,{encoded_image}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": base64_url}},
                {"type": "text", "text": "Extract all text from this image clearly."}
            ]
        }]

        ocr_response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages
        )

        text = ocr_response.choices[0].message.content.strip()
        return jsonify({"ocr_text": text or "", "message": "Success"}), 200

    except Exception as e:
        app.logger.error(f"OCR Error: {e}")
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    if not client:
        return jsonify({"error": "Mistral client not initialized"}), 500

    try:
        data = request.get_json()
        ocr_text = data.get("ocr_text", "")
        user_diseases = data.get("user_diseases", [])
        if not isinstance(user_diseases, list):
            return jsonify({"error": "user_diseases must be a list"}), 400

        diseases_string = ", ".join(user_diseases) or "no known conditions"
        prompt = f"""
You are NoshGuard, a food ingredient analyzer.
User diseases: {diseases_string}
Scanned ingredients:
---
{ocr_text}
---
Analyze possible health risks and give a short summary.
Conclude with: 'This is not medical advice.'
""".strip()

        messages = [{"role": "user", "content": prompt}]

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=0.2
        )

        advice = response.choices[0].message.content.strip()
        return jsonify({"advice": advice}), 200

    except Exception as e:
        app.logger.error(f"Analyze Error: {e}")
        return jsonify({"error": f"Analyze failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
