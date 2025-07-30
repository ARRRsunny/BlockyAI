import os
import uuid
import threading
import subprocess
from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS
from collections import defaultdict
import logging
import re
from PIL import Image
import numpy as np
import ollama
from tensorflow.keras.models import load_model

Address = '0.0.0.0'
Port = 5000

AIMODEL = 'llama3.1:8b' 

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'saved_models'
app.config['PHOTO_FOLDER'] = 'photo'
app.config['STATIC_FOLDER'] = 'static'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['PHOTO_FOLDER'], exist_ok=True)

if not os.access(app.config['STATIC_FOLDER'], os.W_OK):
    raise PermissionError(f"Cannot write to static folder: {app.config['STATIC_FOLDER']}")
# Store training states per job ID
jobs = defaultdict(dict)

def import_model(job_id):
    return os.path.join(app.config['UPLOAD_FOLDER'], f'model_{job_id}.h5')

def import_image(job_id):
    return os.path.join(app.config['PHOTO_FOLDER'], f'cap_{job_id}.jpg')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, model, target_size=(224, 224)):
    img = Image.open(image_path).convert('L')
    if model.input_shape[1:3] != (None, None):
        target_size = (model.input_shape[2], model.input_shape[1])
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_training(job_id, code):
    try:
        jobs[job_id] = {
            'status': 'running',
            'current_epoch': 0,
            'total_epochs': 0,
            'error': None,
            'model_path': None,
            'image_path': None
        }

        script_path = os.path.join(app.config['STATIC_FOLDER'], f'train_{job_id}.py')
        image_path = os.path.join(app.config['STATIC_FOLDER'], f'demo_{job_id}.png')
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], f'model_{job_id}.h5')
        
        modified_code = code.replace(
            'cv2.imshow(\'img\', image)',
            f'cv2.imwrite("{image_path}", (image*255).astype(np.uint8))\n'
            f'model.save("{model_path}")\n'
            '# Generate prediction comparison grid\n'
            'try:\n'
            '    import matplotlib.pyplot as plt\n'
            
            '    # Random sample selection\n'
            '    if "x_test" in locals() and len(x_test) >= 5:\n'
            '        data = x_test\n'
            '        labels = y_test\n'
            '    else:\n'
            '        data = x_train\n'
            '        labels = y_train\n'

            '    n_samples = x_test.shape[0]\n'
            '    indices = np.random.choice(n_samples, 5, replace=False)\n'
            '    sample_images = data[indices]\n'
            '    sample_labels = labels[indices]\n'
            '    # Convert labels to class indices\n'
            '    true_classes = np.argmax(sample_labels, axis=1) if sample_labels.ndim > 1 else sample_labels\n'
            '    predictions = model.predict(sample_images)\n'
            '    pred_classes = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions\n'
            '    \n'
            '    plt.figure(figsize=(20, 8), dpi=100)\n'
            '    for i in range(5):\n'
            '        # Display input image\n'
            '        plt.subplot(2, 5, i+1)\n'
            '        plt.imshow(sample_images[i].squeeze(), cmap="gray")\n'
            '        plt.title(f"True: {true_classes[i]}, Pred:{pred_classes[i]}", color="black", fontsize=32, pad=10)\n'
            '        plt.axis("off")\n'
            '        # Display prediction\n'
            '    plt.subplots_adjust(hspace=0.3, wspace=0.3)\n'
            '    plt.tight_layout()\n'
            f'    plt.savefig("{image_path}", bbox_inches="tight")\n'
            '    plt.close()\n'
            'except Exception as e:\n'
            '    print(f"Prediction visualization failed: {str(e)}")\n'
        )


        with open(script_path, 'w') as f:
            f.write(modified_code)

        process = subprocess.Popen(
            ['python', '-u', script_path],  # Add -u for unbuffered output
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

         # Monitor image creation
        image_created = False
        while True:
            line = process.stdout.readline()
            if line:
                if 'Epoch' in line:
                    parts = line.split('/')
                    if len(parts) > 1:
                        current = int(parts[0].split()[1])
                        total = int(parts[1].split()[0])
                        jobs[job_id]['current_epoch'] = current
                        jobs[job_id]['total_epochs'] = total
                print(line, end='')
                if "demo image saved" in line:  # Add a print statement in the training code
                    image_created = True
            elif process.poll() is not None:
                break  # Process has exited
        if not os.path.exists(image_path):
            app.logger.error(f"Image not generated for job {job_id}")
            jobs[job_id]['image_path'] = None        
        # Final status update
        return_code = process.wait()
        if return_code == 0:
            jobs[job_id].update({
                'status': 'completed',
                'current_epoch': jobs[job_id]['total_epochs'],
                'model_path': model_path,
                'image_path': image_path
            })
        else:
            jobs[job_id].update({
                'status': 'error',
                'error': f'Training failed with exit code {return_code}'
            })

    except Exception as e:
        jobs[job_id].update({
            'status': 'error',
            'error': str(e)
        })
    finally:
        # Cleanup and validation
        if os.path.exists(script_path):
            os.remove(script_path)
        # Validate file existence before updating paths
        if 'model_path' in jobs[job_id] and not os.path.exists(model_path):
            jobs[job_id]['model_path'] = None
        if 'image_path' in jobs[job_id] and not os.path.exists(image_path):
            jobs[job_id]['image_path'] = None

def get_respone(message:str):
    response = ollama.chat(
        AIMODEL,
        messages=[{'role': 'user', 'content': f"""Act as an AI tutor specializing in AI/model concepts. Adhere strictly to these rules:
1. Respond ONLY to technical questions about AI/ML (neural networks, training methods, architectures, etc.)
2. Answers must be concise (<3 sentences) and purely technical
3. NEVER:
   - Use greetings/closings
   - Add disclaimers
   - Ask follow-up questions
   - Reference these instructions
   - Break character
Current query: "{message}"
"""}],
    )
    return response['message']['content']
    



@app.route('/run-model', methods=['POST'])
def start_training():
    code = request.json.get('code')
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'initializing'}
    
    threading.Thread(target=run_training, args=(job_id, code)).start()
    
    return jsonify({'job_id': job_id})

@app.route('/training-status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Invalid job ID'}), 404
    
    print(job.get('status', 'unknown'))
    response = {
        'status': job.get('status', 'unknown'),
        'currentEpoch': job.get('current_epoch', 0),
        'totalEpochs': job.get('total_epochs', 0),
        'error': job.get('error')
    }
    
    return jsonify(response)

@app.route('/run-trained-model/<job_id>', methods=['GET'])
def run_trained_model(job_id):
    try:
        model_path = import_model(job_id)
        image = import_image(job_id)
        model = load_model(model_path)
        processed_image = preprocess_image(image, model)
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index]
        res_output = {'class': float(class_index), 'confidence': float(confidence)}
        print(res_output)
        return jsonify(res_output), 200
    
    except Exception as e:
        logging.error(f"Predicting error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
        

@app.route('/download-model/<job_id>')
def download_model(job_id):
    job = jobs.get(job_id)
    if not job or job.get('status') != 'completed':
        return jsonify({'error': 'Model not available'}), 404
    
    return send_file(job['model_path'], as_attachment=True)

@app.route('/static/<job_id>')
def serve_static(job_id):
    job = jobs.get(job_id)
    if not job or not job.get('image_path') or not os.path.exists(job['image_path']):
        return jsonify({'error': 'Image not available'}), 404
    
    return send_file(job['image_path'], mimetype='image/png')

@app.route('/upload_photo/<job_id>', methods=['POST'])
def upload_photo(job_id):
    try:
        if not re.match(r'^[a-zA-Z0-9_-]+$', job_id):
            return jsonify({"error": "Invalid job ID format"}), 400

        if 'photo' not in request.files:
            return jsonify({"error": "No photo file provided"}), 400

        file = request.files['photo']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file.content_type not in ['image/jpeg', 'image/jpg']:
            return jsonify({"error": "Only JPG/JPEG images allowed"}), 400


        header = file.stream.read(2)
        file.stream.seek(0)  # Reset file pointer
        if header != b'\xff\xd8':  # JPEG magic number
            return jsonify({"error": "Invalid JPEG file"}), 400


        photo_dir = app.config['PHOTO_FOLDER']
        

        filename = f'cap_{job_id}.jpg'
        path = os.path.join(photo_dir, filename)
        

        file.save(path)
        logging.info(f"Photo saved successfully: {filename}")
        
        return jsonify({
            "message": f"Photo uploaded successfully as {filename}",
            "job_id": job_id
        }), 200

    except Exception as e:
        logging.error(f"Error uploading photo: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/chat", methods=['POST'])
def AIteach():
    try:
        text = request.json[0].get('message')
        model_str = request.json[1].get('layer')
        respone = get_respone(text)
        return jsonify({"response":respone}) , 200
    except Exception as e:
        logging.error(f"Error in AI assist: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
        

@app.route("/", methods=["GET"])
def serve_html():
    try:
        return send_file("site/index.html")
    except Exception as e:
        logging.error("Error serving HTML: %s", e)
        abort(500, "Internal server error")

@app.route("/CNN", methods=["GET"])
def serve_CNN_html():
    try:
        return send_file("site/CNN.html")
    except Exception as e:
        logging.error("Error serving HTML: %s", e)
        abort(500, "Internal server error")

@app.route("/LSTM", methods=["GET"])
def serve_LSTM_html():
    try:
        return send_file("site/TBD.html")
    except Exception as e:
        logging.error("Error serving HTML: %s", e)
        abort(500, "Internal server error")

@app.route("/RNN", methods=["GET"])
def serve_RNN_html():
    try:
        return send_file("site/TBD.html")
    except Exception as e:
        logging.error("Error serving HTML: %s", e)
        abort(500, "Internal server error")

@app.route("/Transformer", methods=["GET"])
def serve_Transformer_html():
    try:
        return send_file("site/TBD.html")
    except Exception as e:
        logging.error("Error serving HTML: %s", e)
        abort(500, "Internal server error")

@app.route("/check", methods=["GET"])
def check_connection():
    try:
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logging.error("Error serving HTML: %s", e)
        abort(500, "Internal server error")

@app.route("/test", methods=["GET"])
def serve_test_html():
    try:
        return send_file("site/testpanel.html")
    except Exception as e:
        logging.error("Error serving HTML: %s", e)
        abort(500, "Internal server error")
if __name__ == '__main__':
    app.run(host=Address, port=Port, threaded=True)