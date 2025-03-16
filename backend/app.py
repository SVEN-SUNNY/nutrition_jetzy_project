# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd

app = Flask(__name__)
#CORS(app)  # Enable Cross-Origin Resource Sharing
CORS(app, resources={
    r"/plan": {
        "origins": ["https://jetzy-nutrition-plan.netlify.app"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'nutrition_model.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'feature_encoder.pkl')
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
except Exception as e:
    print(f"Error loading ML artifacts: {str(e)}")
    model = None

def validate_input(data):
    """Validate user input structure"""
    required = ['diet', 'goal']
    if not all(key in data for key in required):
        return False, "Missing required fields"
    if not isinstance(data['diet'], list) or len(data['diet']) == 0:
        return False, "Invalid dietary preferences"
    return True, ""

def predict_plan(user_data):
    """Generate plan using ML model"""
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([{
            'diet': user_data['diet'][0],
            'goal': user_data['goal']
        }])
        
        # Transform features
        encoded = encoder.transform(input_df)
        
        # Predict
        plan_id = model.predict(encoded)[0]
        return get_plan_from_id(plan_id)
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

def get_plan_from_id(plan_id):
    """Get plan details from predefined database"""
    plans = {
        0: {
            'meals': {
                'breakfast': 'Oatmeal with berries',
                'lunch': 'Quinoa salad',
                'dinner': 'Grilled vegetables'
            },
            'calories': 1500,
            'macros': {'protein': 55, 'carbs': 120, 'fats': 40}
        },
        1: {
            'meals': {
                'breakfast': 'Egg white omelette',
                'lunch': 'Grilled chicken',
                'dinner': 'Salmon with broccoli'
            },
            'calories': 2200,
            'macros': {'protein': 110, 'carbs': 150, 'fats': 70}
        }
    }
    return plans.get(plan_id, plans[0])

@app.route('/plan', methods=['POST'])
def generate_plan():
    try:
        data = request.json
        
        # Input validation
        is_valid, msg = validate_input(data)
        if not is_valid:
            return jsonify({'error': msg}), 400
        
        # Generate plan
        if model:
            plan = predict_plan(data)
        else:
            plan = get_plan_from_id(0)  # Fallback
        
        return jsonify({
            'success': True,
            'plan': {
                'meals': plan['meals'],
                'calories': plan['calories'],
                'macros': plan['macros']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
