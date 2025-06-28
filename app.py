from flask import Flask, request, jsonify
import base64
import json
import os
import cv2
import numpy as np
from datetime import datetime
import uuid
import tempfile
from dotenv import load_dotenv
from together import Together
load_dotenv()
app = Flask(__name__)

# Load environment variables
print("API KEyyyyyyyyyyyyyyyyyyyyyyyy: ",os.environ.get("TOGETHER_API_KEY"))

# Create directories for saving images and logs
IMAGES_DIR = "captured_images"
LOGS_DIR = "request_logs"
RESULTS_DIR = "detection_results"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

client = Together()

def get_vision_inference(image_bytes: bytes, prompt: str):
    try:
        # Process image bytes
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Generate stream
        stream = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image;base64,{image_base64}"
                            }
                        },
                    ],
                }
            ],
            stream=True,
        )
        
        # Collect the response
        response_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        
        return response_text.strip()
    
    except Exception as e:
        print(f"Error getting vision inference: {e}")
        return None

def save_image_from_base64(base64_data, filename):
    """Save base64 encoded image to file"""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        
        # Save to file
        filepath = os.path.join(IMAGES_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return filepath, image_data
    except Exception as e:
        print(f"Error saving image: {e}")
        return None, None

def process_tea_image_with_ai(image_bytes):
    try:
        prompt = """Is there a tea leaf with two leaves and one bud in this image? Respond only with:

                    "yes"

                    or

                    "no"

                    No additional explanation. No punctuation. Lowercase only but be strict in analyzing and distinguishing leaves make sure that it is tea leaf cause there is other leaves like that before respond.
                    """
        
        ai_response = get_vision_inference(image_bytes, prompt)
        
        if ai_response:
            return ai_response, None
        else:
            return None, "Failed to get response from Together AI"
        
    except Exception as e:
        print(f"Error processing image with AI: {e}")
        return None, str(e)

def generate_tea_quality_response(ai_response):
    if ai_response is None:
        return "Unable to analyze the tea leaf image. Please ensure the image is clear and contains tea leaves."
    
    return ai_response

def save_analysis_result(image_path, ai_response):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"analysis_result_{timestamp}.txt"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        with open(result_path, 'w') as f:
            f.write(f"Tea Leaf Analysis Result\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Image: {os.path.basename(image_path)}\n")
            f.write(f"AI Analysis:\n{ai_response}\n")
        
        return result_path
    except Exception as e:
        print(f"Error saving analysis result: {e}")
        return None

def log_request(request_data, response_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"request_{timestamp}_{uuid.uuid4().hex[:8]}.json"
    log_filepath = os.path.join(LOGS_DIR, log_filename)
    
    log_entry = {
        "timestamp": timestamp,
        "request": request_data,
        "response": response_data
    }
    
    try:
        with open(log_filepath, 'w') as f:
            json.dump(log_entry, f, indent=2)
    except Exception as e:
        print(f"Error logging request: {e}")

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({"error": "No JSON data received"}), 400
        
        print(f"Received request from ESP32-CAM at {datetime.now()}")
        
        image_saved = False
        image_filename = None
        image_path = None
        image_bytes = None
        
        if 'messages' in request_data:
            for message in request_data['messages']:
                if 'content' in message and isinstance(message['content'], list):
                    for content_item in message['content']:
                        if (content_item.get('type') == 'image_url' and 
                            'image_url' in content_item and 
                            'url' in content_item['image_url']):
                            
                            image_url = content_item['image_url']['url']
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_filename = f"leaf_image_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
                            
                            image_path, image_bytes = save_image_from_base64(image_url, image_filename)
                            if image_path and image_bytes:
                                image_saved = True
                                print(f"Image saved: {image_path}")
                            else:
                                print("Failed to save image")
                            break
        
        response_text = "Unable to process the image. Please ensure the image is clear and contains tea leaves."
        
        if image_saved and image_bytes:
            print("Processing tea leaf analysis")
            ai_response, error = process_tea_image_with_ai(image_bytes)
            
            if error:
                response_text = f"Error analyzing tea leaves: {error}"
            else:
                response_text = generate_tea_quality_response(ai_response)
                
                if ai_response and image_path:
                    result_path = save_analysis_result(image_path, ai_response)
                    if result_path:
                        print(f"Analysis result saved: {result_path}")
        
        # Create response in OpenAI API format
        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "together-ai-vision-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": len(response_text.split()),
                "total_tokens": 100 + len(response_text.split())
            }
        }
        
        # Log the request and response
        log_data = request_data.copy()
        # Remove base64 image data from logs to keep them manageable
        if 'messages' in log_data:
            for message in log_data['messages']:
                if 'content' in message and isinstance(message['content'], list):
                    for content_item in message['content']:
                        if content_item.get('type') == 'image_url':
                            content_item['image_url']['url'] = f"[IMAGE_SAVED_AS_{image_filename}]"
        
        log_request(log_data, response_data)
        
        print(f"Analysis complete. Image saved: {image_saved}")
        print(f"Response: {response_text[:100]}...")
        print("-" * 50)
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing request: {e}")
        error_response = {
            "error": {
                "message": f"Internal server error: {str(e)}",
                "type": "server_error",
                "code": "internal_error"
            }
        }
        return jsonify(error_response), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        test_response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        ai_status = "connected" if test_response else "error"
    except:
        ai_status = "disconnected"
    
    return jsonify({
        "status": "healthy",
        "ai_status": ai_status,
        "timestamp": datetime.now().isoformat(),
        "images_saved": len(os.listdir(IMAGES_DIR)) if os.path.exists(IMAGES_DIR) else 0,
        "results_saved": len(os.listdir(RESULTS_DIR)) if os.path.exists(RESULTS_DIR) else 0
    })

@app.route('/images', methods=['GET'])
def list_images():
    try:
        images = []
        if os.path.exists(IMAGES_DIR):
            for filename in os.listdir(IMAGES_DIR):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(IMAGES_DIR, filename)
                    stat = os.stat(filepath)
                    images.append({
                        "filename": filename,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
        
        return jsonify({
            "total_images": len(images),
            "images": sorted(images, key=lambda x: x['created'], reverse=True)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results', methods=['GET'])
def list_results():
    try:
        results = []
        if os.path.exists(RESULTS_DIR):
            for filename in os.listdir(RESULTS_DIR):
                if filename.lower().endswith('.txt'):
                    filepath = os.path.join(RESULTS_DIR, filename)
                    stat = os.stat(filepath)
                    results.append({
                        "filename": filename,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
        
        return jsonify({
            "total_results": len(results),
            "results": sorted(results, key=lambda x: x['created'], reverse=True)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        image_count = len(os.listdir(IMAGES_DIR)) if os.path.exists(IMAGES_DIR) else 0
        log_count = len(os.listdir(LOGS_DIR)) if os.path.exists(LOGS_DIR) else 0
        result_count = len(os.listdir(RESULTS_DIR)) if os.path.exists(RESULTS_DIR) else 0
        
        return jsonify({
            "total_images": image_count,
            "total_requests": log_count,
            "total_results": result_count,
            "uptime": "Server running",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
   
    app.run(host='0.0.0.0', port=5000, debug=True)