from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime
import os
from meal_plans import MEAL_PLANS, get_rule_based_plan, train_model

app = Flask(__name__)
CORS(app, resources={
    r"/plan": {"origins": ["https://jetzy-nutrition-plan.netlify.app"]},
    r"/selection": {"origins": ["https://jetzy-nutrition-plan.netlify.app"]}
})

# Configuration
SUBMISSIONS_FILE = 'submissions.json'
MODEL_DIR = 'model'

# Configure numpy random generator
from numpy.random import Generator, MT19937
rng = Generator(MT19937(12345))

def store_submission(data):
    """Store user submission with validation"""
    try:
        submission = {
            "timestamp": datetime.now().isoformat(),
            "name": data.get('name', ''),
            "diet": data.get('diet', []),
            "goal": data.get('goal', ''),
            "selected_plan_id": int(data.get('selected_plan_id', -1))
        }
        
        if submission['selected_plan_id'] not in MEAL_PLANS:
            raise ValueError("Invalid plan ID")
            
        with open(SUBMISSIONS_FILE, 'a') as f:
            f.write(json.dumps(submission) + '\n')
            
    except Exception as e:
        app.logger.error(f"Submission storage error: {str(e)}")

def load_model():
    """Load current model and encoder with numpy compatibility fix"""
    try:
        model_path = os.path.join(MODEL_DIR, 'nutrition_model.pk1')
        encoder_path = os.path.join(MODEL_DIR, 'feature_encoder.pk1')
        
        if not all([os.path.exists(model_path), os.path.exists(encoder_path)]):
            raise FileNotFoundError("Model files missing")
            
        return joblib.load(model_path), joblib.load(encoder_path)
        
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}")
        return None, None

@app.route('/plan', methods=['POST'])
def generate_nutrition_plan():
    """Generate personalized nutrition recommendations"""
    try:
        data = request.get_json()
        
        # Validate input
        if not all(key in data for key in ['name', 'diet', 'goal']):
            return jsonify({"success": False, "error": "Missing required fields"}), 400
            
        # Load ML model
        model, encoder = load_model()
        
        if model and encoder:
            # Prepare input features
            diet_value = data['diet'][0] if isinstance(data['diet'], list) else data['diet']
            input_df = pd.DataFrame([{
                'diet': diet_value,
                'goal': data['goal'],
                'diet_goal': f"{diet_value}_{data['goal']}"
            }])
            
            # Generate predictions
            encoded = encoder.transform(input_df)
            proba = model.predict_proba(encoded)[0]
            top5_idx = np.argsort(proba)[-5:][::-1]
            
            plans = [{
                **MEAL_PLANS[model.classes_[i]],
                "confidence": float(proba[i])
            } for i in top5_idx]
        else:
            plans = [get_rule_based_plan(data)]
            
        return jsonify({"success": True, "plans": plans})
        
    except Exception as e:
        app.logger.error(f"Plan generation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to generate plan",
            "fallback": get_rule_based_plan(data)
        }), 500

@app.route('/selection', methods=['POST'])
def handle_plan_selection():
    """Process user plan selection and retrain model"""
    try:
        data = request.get_json()
        
        # Validate selection
        if 'selected_plan_id' not in data or data['selected_plan_id'] not in MEAL_PLANS:
            return jsonify({"success": False, "error": "Invalid plan selection"}), 400
            
        # Store submission
        store_submission(data)
        
        # Retrain model
        try:
            train_model()
            return jsonify({"success": True})
        except Exception as e:
            app.logger.error(f"Model retraining failed: {str(e)}")
            return jsonify({"success": False, "error": "Selection saved but model update failed"}), 500
            
    except Exception as e:
        app.logger.error(f"Selection processing error: {str(e)}")
        return jsonify({"success": False, "error": "Failed to process selection"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(os.path.join(MODEL_DIR, 'nutrition_model.pk1')),
        "numpy_version": np.__version__,
        "sklearn_version": joblib.__version__
    })

def initialize_system():
    """Initialize application components"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Train initial model if missing
    if not os.path.exists(os.path.join(MODEL_DIR, 'nutrition_model.pk1')):
        try:
            train_model()
            app.logger.info("Initial model trained successfully")
        except Exception as e:
            app.logger.error(f"Initial model training failed: {str(e)}")
            raise

if __name__ == '__main__':
    initialize_system()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
