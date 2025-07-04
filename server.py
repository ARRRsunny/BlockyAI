import os
import uuid
import threading
import subprocess
from flask import Flask, jsonify, request, send_file,   abort
from flask_cors import CORS
from collections import defaultdict
import logging

Address = '0.0.0.0'
Port = 5000

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'saved_models'
app.config['PHOTO_FOLDER'] = 'photo'
app.config['STATIC_FOLDER'] = 'static'
app.config['photo_path'] = 'photo/cap.jpg'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['PHOTO_FOLDER'], exist_ok=True)

if not os.access(app.config['STATIC_FOLDER'], os.W_OK):
    raise PermissionError(f"Cannot write to static folder: {app.config['STATIC_FOLDER']}")
# Store training states per job ID
jobs = defaultdict(dict)

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

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    try:
        if 'photo' not in request.files:
            return jsonify({"error": "No photo file provided."}), 400

        file = request.files['photo']
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400

        file.save(app.config['photo_path'])
        try:
            print('saved')
        except Exception as e:
            logging.error(f"Error YOLO cropping: {e}")
            
        return jsonify({"message": "Photo uploaded successfully and replaced cap.jpg."}), 200

    except Exception as e:
        logging.error(f"Error uploading photo: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def serve_html():
    try:
        return send_file("index.html")
    except Exception as e:
        logging.error("Error serving HTML: %s", e)
        abort(500, "Internal server error")

if __name__ == '__main__':
    app.run(host=Address, port=Port, threaded=True)