from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from better_running_inference import infer_chords, format_time

# Initialize Flask app
app = Flask(__name__, static_folder='frontend/build')
CORS(app)  # Enable CORS for all routes

# API endpoints
@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_audio.mp3")
        file.save(temp_path)
        
        # Process the audio file
        model_weights_path = "pcpmodel_1000_regular_learning.h5"
        chord_segments = infer_chords(temp_path, model_weights_path)
        
        # Format the results for frontend
        results = [
            {
                'startTime': start_time,
                'endTime': end_time,
                'chord': chord
            }
            for start_time, end_time, chord in chord_segments
        ]
        
        # Clean up
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            'success': True,
            'chordData': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Serve React static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)