from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
from mistralai import Mistral # Use 'Mistral' from the main package

app = Flask(__name__)
# This enables CORS for all domains. For production, you might want to restrict it:
# CORS(app, resources={r"/*": {"origins": "http://yourinfinityfreedomain.infinityfreeapp.com"}})
CORS(app)

# --- API KEY SECTION ---
# The API key will be set as an environment variable on Render
api_key = "aCHLMXr7v1Ojf2lniUNkjftHGnzxLoDp" # YOUR API KEY
# YOUR API KEY

if not api_key:
    print("CRITICAL ERROR: MISTRAL_API_KEY environment variable not set.")
    # You might want to raise an exception or exit if the API key is crucial at startup
    # For now, it will allow the app to start, but API calls will fail.
    # client initialization will likely fail below if api_key is None.

# Initialize the Mistral client
try:
    if api_key:
        client = Mistral(api_key=api_key)
        print("Mistral client initialized successfully.")
    else:
        # Handle case where API key is missing, client cannot be initialized
        client = None # Or raise an error
        print("Mistral client NOT initialized due to missing API key.")
except Exception as e:
    print(f"Error initializing Mistral client: {e}")
    client = None # Ensure client is None if initialization fails
    # Optionally, re-raise or exit if client initialization is critical
    # raise

@app.route('/')
def home():
    return jsonify({"message": "NoshGuard Python Backend is running!"}), 200

@app.route('/ocr', methods=['POST'])
def run_ocr():
    if not client:
        return jsonify({"error": "Mistral client not initialized. Check API key."}), 500

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # Read image bytes directly from the uploaded file
        image_bytes = file.read()
        encoded_image_string = base64.b64encode(image_bytes).decode('utf-8')
        
        # Use the mimetype provided by the client/browser for the data URL
        base64_data_url = f"data:{file.mimetype};base64,{encoded_image_string}"

        messages_for_ocr = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_data_url}},
                    {"type": "text", "text": "Extract all text from this image. Present it clearly and concisely."}
                ]
            }
        ]
        
        ocr_response = client.chat.complete(
            model="mistral-small-latest", # Or your preferred model
            messages=messages_for_ocr
        )
        
        extracted_text = ocr_response.choices[0].message.content.strip()

        # If OCR is successful but no text is found, return 200 OK with empty text
        if not extracted_text:
            return jsonify({"ocr_text": "", "message": "OCR process completed, but no text was extracted from the image."}), 200
        
        return jsonify({"ocr_text": extracted_text}), 200

    except Exception as e:
        app.logger.error(f"OCR Error: {str(e)}")
        return jsonify({"error": f"OCR processing error: {str(e)}"}), 500

@app.route('/analyze', methods=['POST'])
def analyze_health_risk():
    if not client:
        return jsonify({"error": "Mistral client not initialized. Check API key."}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        ocr_text = data.get("ocr_text", "") # Default to empty string if not provided
        user_diseases = data.get("user_diseases", [])

        # No need to error if ocr_text is empty, the prompt can handle it.
        # if not ocr_text and ocr_text != "": # This condition was a bit complex
        #     return jsonify({"error": "Missing 'ocr_text'"}), 400

        if not isinstance(user_diseases, list):
            return jsonify({"error": "'user_diseases' must be a list"}), 400

        diseases_string = ", ".join(user_diseases) if user_diseases else "no specific medical conditions stated"
        
        prompt_text_content = f"""You are NoshGuard, a health-conscious food ingredient analyzer.
User's conditions: {diseases_string}.
Scanned ingredients:
---
{ocr_text}
---
Analyze potential risks based *only* on the ingredients and conditions. If risks exist, explain simply. If none are obvious, state that. Conclude with a general suitability remark (e.g., "likely suitable," "caution advised," "may not be suitable").
**CRITICAL: Always state: 'This is not medical advice. Consult a doctor or nutritionist for personalized guidance.'**
Be concise (under 100 words). Do not use markdown formatting.
"""
        
        messages_for_analysis = [
            {"role": "user", "content": prompt_text_content}
        ]

        response = client.chat.complete(
            model="mistral-small-latest", # Or your preferred model
            messages=messages_for_analysis,
            temperature=0.2
        )
        
        advice = response.choices[0].message.content.strip()
        return jsonify({"advice": advice}), 200

    except Exception as e:
        app.logger.error(f"Analysis Error: {str(e)}")
        return jsonify({"error": f"Analysis processing error: {str(e)}"}), 500

if __name__ == '__main__':
    # Gunicorn will be used by Render to run the app.
    # This block is mainly for local development/testing.
    # Render provides the PORT environment variable.
    port = int(os.environ.get("PORT", 5001)) # Default to 5001 for local if not set
    # Listen on 0.0.0.0 to be accessible externally if needed for local testing
    app.run(host='0.0.0.0', port=port, debug=False) # debug=False for production
