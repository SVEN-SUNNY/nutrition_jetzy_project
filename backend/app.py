from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={
    r"/plan": {
        "origins": ["https://jetzy-nutrition-plan.netlify.app"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Load ML model and encoder
try:
    model = joblib.load('model/nutrition_model.pk1')
    encoder = joblib.load('model/feature_encoder.pk1')
except Exception as e:
    print(f"Error loading ML artifacts: {str(e)}")
    model = None

# Meal Plans Database
MEAL_PLANS = {
    0: {
        'name': 'Vegetarian Weight Loss',
        'meals': {
            'breakfast': 'Oatmeal with berries and chia seeds',
            'lunch': 'Quinoa salad with roasted vegetables',
            'dinner': 'Lentil curry with brown rice'
        },
        'calories': 1500,
        'macros': {
            'breakfast': {'calories': 350, 'protein': 12},
            'lunch': {'calories': 450, 'protein': 18},
            'dinner': {'calories': 700, 'protein': 25},
            'daily': {'protein': 55, 'carbs': 180, 'fats': 40}
        }
    },
    1: {
        'name': 'High Protein Muscle Gain',
        'meals': {
            'breakfast': 'Egg white omelette with spinach',
            'lunch': 'Grilled chicken with sweet potato',
            'dinner': 'Salmon with asparagus'
        },
        'calories': 2500,
        'macros': {
            'breakfast': {'calories': 500, 'protein': 30},
            'lunch': {'calories': 700, 'protein': 50},
            'dinner': {'calories': 1300, 'protein': 70},
            'daily': {'protein': 150, 'carbs': 200, 'fats': 70}
        }
    }
}

@app.route('/plan', methods=['POST'])
def get_plan():
    try:
        # Log incoming request
        print("Received request:", request.json)
        
        # Validate input
        data = request.json
        if not data.get('diet') or not data.get('goal'):
            return jsonify({
                "success": False,
                "error": "Missing required fields"
            }), 400
        
        # Generate plan
        if model:
            # Prepare input
            input_df = pd.DataFrame([{
                'diet': data['diet'][0],
                'goal': data['goal']
            }])
            
            # Encode features
            encoded = encoder.transform(input_df)
            
            # Predict plan
            plan_id = model.predict(encoded)[0]
            plan = MEAL_PLANS[plan_id]
        else:
            plan = get_rule_based_plan(data)
        
        # Log generated plan
        print("Generated plan:", plan)
        
        return jsonify({
            "success": True,
            "plan": plan
        })
    
    except Exception as e:
        # Log error
        print("Error:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def get_rule_based_plan(data):
    """Fallback rule-based plan"""
    diet = data['diet'][0] if data['diet'] else 'vegetarian'
    goal = data['goal']
    
    if diet == 'vegetarian' and goal == 'weight-loss':
        return MEAL_PLANS[0]
    return MEAL_PLANS[1]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
