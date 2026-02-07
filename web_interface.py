import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plant_disease_detector import disease_solutions

# Optimize TensorFlow memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.set_soft_device_placement(True)

# Set memory growth to prevent OOM errors
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

app = Flask(__name__, template_folder='templates')

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'detector_disease.keras')
model = None
IMG_SIZE = (240, 240)


class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


def load_model():
    global model
    try:
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(__file__)}")
        print(f"Looking for model at: {MODEL_PATH}")
        print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        
        # List files in models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if os.path.exists(models_dir):
            print(f"Files in models directory: {os.listdir(models_dir)}")
        else:
            print(f"Models directory does not exist at: {models_dir}")
        
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully! Classes: {len(class_names)}")
            return True
        else:
            print(f"Model file not found at: {MODEL_PATH}")
            # Try alternative path
            alt_path = 'models/detector_disease.keras'
            if os.path.exists(alt_path):
                print(f"Found model at alternative path: {alt_path}")
                model = tf.keras.models.load_model(alt_path)
                print(f"Model loaded from alternative path!")
                return True
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/favicon.ico')
def favicon():
    try:
        favicon_path = os.path.join(app.root_path, 'static', 'favicon.ico')
        if os.path.exists(favicon_path):
            return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    except Exception as e:
        print(f"Favicon error: {e}")
    from flask import Response
    return Response(status=204)

@app.route('/health')
def health_check():
    """Health check endpoint for deployment platforms"""
    if model is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'classes': len(class_names)
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Model not loaded'
        }), 503

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'loaded': False,
            'error': 'Model not loaded'
        })
    
    return jsonify({
        'loaded': True,
        'classes': class_names,
        'num_classes': len(class_names),
        'image_size': IMG_SIZE
    })

@app.route('/diseases')
def get_diseases():
    diseases = []
    for disease_name, info in disease_solutions.items():
        diseases.append({
            'name': disease_name,
            'symptoms': info.get('symptoms', ''),
            'causes': info.get('causes', ''),
            'solutions': info.get('solutions', []),
            'prevention': info.get('prevention', '')
        })
    return jsonify({'diseases': diseases})

@app.route('/predict', methods=['POST'])
@app.route('/upload', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check if detector_disease.keras exists in models folder.'
        }), 500
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")
        
        # Preprocess image
        print("Preprocessing image...")
        img = preprocess_image(filepath)
        if img is None:
            print("ERROR: Image preprocessing failed")
            return jsonify({
                'success': False,
                'error': 'Could not read image. Please upload a valid image file.'
            }), 400
        
        print(f"Image preprocessed successfully. Shape: {img.shape}")
        
        # Make prediction with optimized settings
        print("Making prediction...")
        # Use smaller batch size and disable eager execution for memory efficiency
        with tf.device('/CPU:0'):  # Force CPU to avoid GPU memory issues
            predictions = model.predict(img, verbose=0, batch_size=1)
        print(f"Predictions shape: {predictions.shape}")
        
        # Validate predictions
        if not isinstance(predictions, np.ndarray):
            return jsonify({
                'success': False,
                'error': 'Invalid prediction output from model'
            }), 500
        
        if len(predictions) == 0 or len(predictions[0]) != len(class_names):
            return jsonify({
                'success': False,
                'error': f'Prediction shape mismatch. Got {predictions.shape}, expected (1, {len(class_names)})'
            }), 500
        
        # Get top prediction
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        predicted_class = class_names[class_idx]
        print(f"Predicted: {predicted_class} with confidence: {confidence:.4f}")
        
        # Get top 5 predictions
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_predictions = [
            {   
                'disease': class_names[idx].replace('___', ' - ').replace('_', ' '),
                'confidence': float(predictions[0][idx])
            }
            for idx in top_5_idx
        ]
        
        # Confidence warning system
        confidence_warning = None
        confidence_level = "High"
        
        if confidence < 0.5:
            confidence_warning = "⚠️ LOW CONFIDENCE: This prediction has low confidence. Please verify the diagnosis visually or consult a professional."
            confidence_level = "Low"
        elif confidence < 0.7:
            confidence_warning = "⚠️ MODERATE CONFIDENCE: The prediction could be one of several similar diseases. Check the alternative predictions below."
            confidence_level = "Moderate"
        elif confidence < 0.85:
            confidence_level = "Good"
        else:
            confidence_level = "High"
        
        # Get disease information
        disease_info = disease_solutions.get(predicted_class, {
            "symptoms": "Information not available for this disease",
            "causes": "Information not available",
            "solutions": [
                "Consult with a plant pathologist",
                "Take clear photos of symptoms",
                "Monitor plant health regularly"
            ],
            "prevention": "Regular monitoring and proper plant care"
        })
        
        disease_display = predicted_class.replace('___', ' - ').replace('_', ' ')
        
        from flask import url_for
        image_url = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify({
            'success': True,
            'prediction': {
                'disease': disease_display,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'raw_name': predicted_class
            },
            'confidence_warning': confidence_warning,
            'top_predictions': top_predictions,
            'disease_info': disease_info,
            'image_path': image_url
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Handle user feedback on predictions"""
    try:
        data = request.json
        helpful = data.get('helpful')
        disease = data.get('disease')
        timestamp = data.get('timestamp')
        
        # Create feedback log directory if it doesn't exist
        feedback_dir = os.path.join(os.path.dirname(__file__), 'feedback_logs')
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Log feedback to a file
        feedback_file = os.path.join(feedback_dir, 'feedback.log')
        feedback_entry = f"{timestamp} | Helpful: {helpful} | Disease: {disease}\n"
        
        with open(feedback_file, 'a') as f:
            f.write(feedback_entry)
        
        print(f"Feedback recorded: {feedback_entry.strip()}")
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback!'
        }), 200
    
    except Exception as e:
        print(f"Error recording feedback: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to record feedback'
        }), 500

@app.route('/.well-known/<path:filename>')
def well_known(filename):
    """Handle .well-known requests (DevTools, etc.) silently"""
    return '', 204

@app.errorhandler(404)
def handle_404(e):
    """Handle 404 errors"""
    # Silently ignore certain browser requests
    if request.path.startswith('/.well-known/'):
        return '', 204
    # Return JSON 404 for API requests
    if request.path.startswith('/api/') or request.accept_mimetypes.accept_json:
        return jsonify({'success': False, 'error': 'Not found'}), 404
    # Return HTML 404 for page requests
    return render_template('index.html'), 404

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    import traceback
    error_trace = traceback.format_exc()
    print(f"Unhandled exception: {error_trace}")
    return jsonify({
        'success': False,
        'error': str(e)
    }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    # Load model on startup
    if load_model():
        print("✓ Model loaded successfully!")
    else:
        print("✗ Failed to load model!")
        print(f"✗ Please ensure {MODEL_PATH} exists")

    print("Server running at: http://localhost:5000")
    # Get port from environment variable for deployment platforms
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

# Load model on import for production servers
load_model()