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
        
        # Validate plan ID
        if submission['selected_plan_id'] not in MEAL_PLANS:
            raise ValueError("Invalid plan ID")
            
        with open(SUBMISSIONS_FILE, 'a') as f:
            f.write(json.dumps(submission) + '\n')
            
    except Exception as e:
        app.logger.error(f"Submission storage error: {str(e)}")

def load_model():
    """Load current model and encoder with error handling"""
    try:
        # Configure numpy for compatibility
        np.random.bit_generator = np.random.MT19937()
        
        model_path = os.path.join(MODEL_DIR, 'nutrition_model.pkl')
        encoder_path = os.path.join(MODEL_DIR, 'feature_encoder.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError("Model files missing")
            
        return (
            joblib.load(model_path),
            joblib.load(encoder_path)
        )
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}")
        return None, None

@app.route('/plan', methods=['POST'])
def generate_nutrition_plan():
    """Generate personalized nutrition recommendations"""
    try:
        data = request.json
        app.logger.info(f"Plan request: {data}")
        
        # Validate input
        required_fields = ['name', 'diet', 'goal']
        if not all(data.get(field) for field in required_fields):
            return jsonify({
                "success": False,
                "error": "Missing required fields: name, diet, or goal"
            }), 400
            
        # Load model components
        model, encoder = load_model()
        
        if model and encoder:
            # Prepare features with proper data types
            diet_value = data['diet'][0] if isinstance(data['diet'], list) else data['diet']
            diet_goal = f"{diet_value}_{data['goal']}"
            
            input_df = pd.DataFrame([{
                'diet': diet_value,
                'goal': data['goal'],
                'diet_goal': diet_goal
            }])
            
            # Generate predictions
            encoded = encoder.transform(input_df)
            probabilities = model.predict_proba(encoded)[0]
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            
            plans = [{
                **MEAL_PLANS[model.classes_[i]],
                "confidence": float(probabilities[i])
            } for i in top5_indices]
        else:
            # Fallback to rule-based system
            plans = [get_rule_based_plan(data)]
            
        return jsonify({
            "success": True,
            "plans": plans
        })
        
    except Exception as e:
        app.logger.error(f"Plan generation error: {str(e)}")
        try:
            # Secondary fallback attempt
            return jsonify({
                "success": True,
                "plans": [get_rule_based_plan(data)]
            })
        except Exception as fallback_error:
            return jsonify({
                "success": False,
                "error": f"Main: {str(e)}, Fallback: {str(fallback_error)}"
            }), 500

@app.route('/selection', methods=['POST'])
def handle_plan_selection():
    """Process user plan selection and retrain model"""
    try:
        data = request.json
        app.logger.info(f"Plan selection: {data}")
        
        # Validate selection
        if 'selected_plan_id' not in data or data['selected_plan_id'] not in MEAL_PLANS:
            return jsonify({
                "success": False,
                "error": "Invalid plan selection"
            }), 400
            
        # Store submission
        store_submission(data)
        
        # Retrain model with new data
        try:
            train_model()
            app.logger.info("Model retrained successfully")
            return jsonify({"success": True})
        except Exception as e:
            app.logger.error(f"Model retraining failed: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Selection saved but model update failed"
            }), 500
            
    except Exception as e:
        app.logger.error(f"Selection processing error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to process selection"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """System health monitoring endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": os.path.exists(os.path.join(MODEL_DIR, 'nutrition_model.pkl')),
        "submissions_count": sum(1 for _ in open(SUBMISSIONS_FILE)) if os.path.exists(SUBMISSIONS_FILE) else 0
    }
    return jsonify(status), 200

def initialize_system():
    """Initialize application components"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Train initial model if missing
    if not os.path.exists(os.path.join(MODEL_DIR, 'nutrition_model.pkl')):
        app.logger.info("No model found - training initial model")
        try:
            train_model()
        except Exception as e:
            app.logger.error(f"Initial model training failed: {str(e)}")
            raise

if __name__ == '__main__':
    initialize_system()
    app.run(host='0.0.0.0', port=5000)
