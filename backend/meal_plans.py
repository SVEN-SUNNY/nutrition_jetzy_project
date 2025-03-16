import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


MEAL_PLANS = {
    0: {
        'id': 0,
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
        'id': 1,
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
    },
    2: {
        'id': 2,
        'name': 'Low Carb Maintenance',
        'meals': {
            'breakfast': 'Avocado egg bake',
            'lunch': 'Zucchini noodle chicken stir-fry',
            'dinner': 'Cauliflower crust pizza'
        },
        'calories': 1800,
        'macros': {
            'breakfast': {'calories': 400, 'protein': 25},
            'lunch': {'calories': 500, 'protein': 35},
            'dinner': {'calories': 900, 'protein': 40},
            'daily': {'protein': 100, 'carbs': 50, 'fats': 120}
        }
    },
    3: {
        'id': 3,
        'name': 'Vegan Weight Loss',
        'meals': {
            'breakfast': 'Chia seed pudding',
            'lunch': 'Kale and roasted chickpea salad',
            'dinner': 'Tofu stir-fry with quinoa'
        },
        'calories': 1400,
        'macros': {
            'breakfast': {'calories': 300, 'protein': 10},
            'lunch': {'calories': 400, 'protein': 15},
            'dinner': {'calories': 700, 'protein': 20},
            'daily': {'protein': 45, 'carbs': 160, 'fats': 35}
        }
    },
    4: {
        'id': 4,
        'name': 'Athlete Performance',
        'meals': {
            'breakfast': 'Protein pancakes',
            'lunch': 'Turkey burger with sweet potato fries',
            'dinner': 'Grilled steak with wild rice'
        },
        'calories': 3000,
        'macros': {
            'breakfast': {'calories': 600, 'protein': 40},
            'lunch': {'calories': 800, 'protein': 60},
            'dinner': {'calories': 1600, 'protein': 80},
            'daily': {'protein': 180, 'carbs': 300, 'fats': 80}
        }
    },
    5: {
        'id': 5,
        'name': 'Keto Weight Loss',
        'meals': {
            'breakfast': 'Bulletproof coffee with avocado',
            'lunch': 'Cauliflower rice chicken bowl',
            'dinner': 'Zucchini noodles with pesto and shrimp'
        },
        'calories': 1600,
        'macros': {
            'breakfast': {'calories': 400, 'protein': 20},
            'lunch': {'calories': 500, 'protein': 35},
            'dinner': {'calories': 700, 'protein': 45},
            'daily': {'protein': 100, 'carbs': 30, 'fats': 120}
        }
    },
    6: {
        'id': 6,
        'name': 'Mediterranean Heart Health',
        'meals': {
            'breakfast': 'Greek yogurt with walnuts and honey',
            'lunch': 'Grilled fish with olive tapenade',
            'dinner': 'Chickpea stew with whole grain pita'
        },
        'calories': 1800,
        'macros': {
            'breakfast': {'calories': 400, 'protein': 25},
            'lunch': {'calories': 600, 'protein': 35},
            'dinner': {'calories': 800, 'protein': 40},
            'daily': {'protein': 100, 'carbs': 150, 'fats': 70}
        }
    },
    7: {
        'id': 7,
        'name': 'Pescatarian Muscle Gain',
        'meals': {
            'breakfast': 'Smoked salmon omelette',
            'lunch': 'Tuna steak with quinoa',
            'dinner': 'Shrimp stir-fry with brown rice'
        },
        'calories': 2400,
        'macros': {
            'breakfast': {'calories': 500, 'protein': 35},
            'lunch': {'calories': 700, 'protein': 50},
            'dinner': {'calories': 1200, 'protein': 65},
            'daily': {'protein': 150, 'carbs': 200, 'fats': 80}
        }
    },
    8: {
        'id': 8,
        'name': 'Gluten-Free Maintenance',
        'meals': {
            'breakfast': 'Buckwheat pancakes',
            'lunch': 'Stuffed bell peppers with ground turkey',
            'dinner': 'Baked chicken with mashed cauliflower'
        },
        'calories': 2000,
        'macros': {
            'breakfast': {'calories': 450, 'protein': 30},
            'lunch': {'calories': 600, 'protein': 40},
            'dinner': {'calories': 950, 'protein': 50},
            'daily': {'protein': 120, 'carbs': 120, 'fats': 90}
        }
    },
    9: {
        'id': 9,
        'name': 'Dairy-Free Energy Boost',
        'meals': {
            'breakfast': 'Chia pudding with almond milk',
            'lunch': 'Chicken salad with tahini dressing',
            'dinner': 'Beef curry with coconut milk'
        },
        'calories': 1900,
        'macros': {
            'breakfast': {'calories': 350, 'protein': 25},
            'lunch': {'calories': 500, 'protein': 35},
            'dinner': {'calories': 1050, 'protein': 50},
            'daily': {'protein': 110, 'carbs': 100, 'fats': 100}
        }
    },
    10: {
        'id': 10,
        'name': 'Paleo Athletic Performance',
        'meals': {
            'breakfast': 'Sweet potato hash with eggs',
            'lunch': 'Grilled steak salad',
            'dinner': 'Bison burgers with sweet potato fries'
        },
        'calories': 2800,
        'macros': {
            'breakfast': {'calories': 600, 'protein': 40},
            'lunch': {'calories': 800, 'protein': 55},
            'dinner': {'calories': 1400, 'protein': 75},
            'daily': {'protein': 170, 'carbs': 150, 'fats': 110}
        }
    },
    11: {
        'id': 11,
        'name': 'Low-Fat General Health',
        'meals': {
            'breakfast': 'Oat bran with fruit',
            'lunch': 'Turkey chili',
            'dinner': 'Baked cod with steamed vegetables'
        },
        'calories': 1700,
        'macros': {
            'breakfast': {'calories': 300, 'protein': 20},
            'lunch': {'calories': 450, 'protein': 35},
            'dinner': {'calories': 950, 'protein': 45},
            'daily': {'protein': 100, 'carbs': 180, 'fats': 35}
        }
    },
    12: {
        'id': 12,
        'name': 'High-Fiber Digestive Health',
        'meals': {
            'breakfast': 'Bran cereal with berries',
            'lunch': 'Lentil soup with whole grain bread',
            'dinner': 'Roasted vegetable quinoa bowl'
        },
        'calories': 1850,
        'macros': {
            'breakfast': {'calories': 350, 'protein': 20},
            'lunch': {'calories': 500, 'protein': 25},
            'dinner': {'calories': 1000, 'protein': 40},
            'daily': {'protein': 85, 'carbs': 220, 'fats': 50}
        }
    },
    13: {
        'id': 13,
        'name': 'Plant-Based Endurance',
        'meals': {
            'breakfast': 'Tofu scramble',
            'lunch': 'Black bean Buddha bowl',
            'dinner': 'Tempeh stir-fry with brown rice'
        },
        'calories': 2300,
        'macros': {
            'breakfast': {'calories': 450, 'protein': 25},
            'lunch': {'calories': 650, 'protein': 35},
            'dinner': {'calories': 1200, 'protein': 45},
            'daily': {'protein': 105, 'carbs': 300, 'fats': 60}
        }
    },
    14: {
        'id': 14,
        'name': 'Balanced Family Meals',
        'meals': {
            'breakfast': 'Whole grain waffles with fruit',
            'lunch': 'Chicken fajita bowl',
            'dinner': 'Salmon pasta primavera'
        },
        'calories': 2100,
        'macros': {
            'breakfast': {'calories': 400, 'protein': 25},
            'lunch': {'calories': 600, 'protein': 35},
            'dinner': {'calories': 1100, 'protein': 50},
            'daily': {'protein': 110, 'carbs': 200, 'fats': 70}
        }
    },
    15: {
        'id': 15,
        'name': 'Senior Health Plan',
        'meals': {
            'breakfast': 'Oatmeal with almonds',
            'lunch': 'Grilled fish with veggies',
            'dinner': 'Turkey meatloaf'
        },
        'calories': 1800,
        'macros': {
            'breakfast': {'calories': 350, 'protein': 20},
            'lunch': {'calories': 500, 'protein': 35},
            'dinner': {'calories': 950, 'protein': 35},
            'daily': {'protein': 90, 'carbs': 150, 'fats': 60}
        }
    },
    16: {
        'id': 16,
        'name': 'Pregnancy Nutrition Plan',
        'meals': {
            'breakfast': 'Greek yogurt parfait',
            'lunch': 'Spinach and cheese quesadilla',
            'dinner': 'Grilled salmon with quinoa'
        },
        'calories': 2200,
        'macros': {
            'breakfast': {'calories': 400, 'protein': 25},
            'lunch': {'calories': 600, 'protein': 35},
            'dinner': {'calories': 1200, 'protein': 40},
            'daily': {'protein': 100, 'carbs': 200, 'fats': 70}
        }
    },
    17: {
        'id': 17,
        'name': 'Diabetes-Friendly Plan',
        'meals': {
            'breakfast': 'Scrambled eggs with whole wheat toast',
            'lunch': 'Grilled chicken with veggies',
            'dinner': 'Baked fish with roasted Brussels sprouts'
        },
        'calories': 1800,
        'macros': {
            'breakfast': {'calories': 350, 'protein': 25},
            'lunch': {'calories': 500, 'protein': 35},
            'dinner': {'calories': 950, 'protein': 30},
            'daily': {'protein': 90, 'carbs': 150, 'fats': 60}
        }
    },
    18: {
        'id': 18,
        'name': 'High-Calorie Mass Gain',
        'meals': {
            'breakfast': 'Peanut butter smoothie',
            'lunch': 'Beef burrito bowl',
            'dinner': 'Steak with mashed potatoes'
        },
        'calories': 3500,
        'macros': {
            'breakfast': {'calories': 800, 'protein': 50},
            'lunch': {'calories': 1200, 'protein': 70},
            'dinner': {'calories': 1500, 'protein': 80},
            'daily': {'protein': 200, 'carbs': 300, 'fats': 150}
        }
    },
    19: {
        'id': 19,
        'name': 'Budget-Friendly Nutrition',
        'meals': {
            'breakfast': 'Banana oatmeal',
            'lunch': 'Rice and beans',
            'dinner': 'Vegetable stir-fry with tofu'
        },
        'calories': 1800,
        'macros': {
            'breakfast': {'calories': 300, 'protein': 15},
            'lunch': {'calories': 500, 'protein': 25},
            'dinner': {'calories': 1000, 'protein': 40},
            'daily': {'protein': 80, 'carbs': 200, 'fats': 50}
        }
    }
}

DIET_PREFERENCES = [
    'vegetarian', 'vegan', 'high-protein', 'low-carb', 'keto',
    'mediterranean', 'pescatarian', 'gluten-free', 'dairy-free',
    'paleo', 'low-fat', 'high-fiber', 'plant-based', 'diabetes-friendly',
    'budget-friendly'
]

HEALTH_GOALS = [
    'weight-loss', 'muscle-gain', 'general-health', 'heart-health',
    'athletic-performance', 'pregnancy', 'senior-health', 'mass-gain',
    'digestive-health'
]

def create_synthetic_data() -> pd.DataFrame:
    """Generate realistic synthetic training data"""
    np.random.seed(42)
    size = 5000
    data = {
        'diet': [],
        'goal': [],
        'selected_plan_id': []
    }

    for _ in range(size):
        age_group = np.random.choice(['young', 'adult', 'senior'], p=[0.3, 0.5, 0.2])
        goal = np.random.choice(HEALTH_GOALS)
        
        # Diet selection logic
        if goal == 'muscle-gain':
            diet = np.random.choice(['high-protein', 'paleo', 'keto'])
        elif goal == 'weight-loss':
            diet = np.random.choice(['low-carb', 'keto', 'mediterranean'])
        elif goal == 'senior-health':
            diet = np.random.choice(['mediterranean', 'low-fat', 'plant-based'])
        else:
            diet = np.random.choice(DIET_PREFERENCES)

        # Plan selection rules
        plan_rules = {
            ('high-protein', 'muscle-gain'): [1, 4, 18],
            ('low-carb', 'weight-loss'): [2, 5, 17],
            ('mediterranean', 'heart-health'): [6, 15],
            ('diabetes-friendly', 'general-health'): [17],
            ('senior-health', 'senior-health'): [15],
            ('pregnancy', 'pregnancy'): [16],
            ('budget-friendly', None): [19]
        }

        matched_plans = []
        for (d, g), ids in plan_rules.items():
            if d == diet and (g == goal or g is None):
                matched_plans.extend(ids)
        
        plan_id = np.random.choice(matched_plans) if matched_plans else np.random.choice(list(MEAL_PLANS.keys()))
        
        data['diet'].append(diet)
        data['goal'].append(goal)
        data['selected_plan_id'].append(plan_id)

    return pd.DataFrame(data)

def load_user_submissions() -> pd.DataFrame:
    """Load and validate real user submissions"""
    try:
        submissions = pd.read_json('submissions.json', lines=True)
        valid_submissions = submissions[
            (submissions['selected_plan_id'].between(0, 19)) &
            (submissions['diet'].apply(lambda x: x[0] in DIET_PREFERENCES)) &
            (submissions['goal'].isin(HEALTH_GOALS))
        ]
        return valid_submissions[['diet', 'goal', 'selected_plan_id']]
    except Exception as e:
        return pd.DataFrame()

def create_dataset() -> pd.DataFrame:
    """Combine synthetic and real data"""
    synthetic = create_synthetic_data()
    real = load_user_submissions()
    return pd.concat([synthetic, real], ignore_index=True)

def train_model() -> None:
    """Train and optimize the recommendation model"""
    try:
        os.makedirs('model', exist_ok=True)
        df = create_dataset()
        
        # Feature engineering
        df['diet_goal'] = df['diet'] + "_" + df['goal']
        
        # Advanced encoding
        encoder = OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            max_categories=50,
            sparse_output=False
        )
        
        encoded_features = encoder.fit_transform(df[['diet', 'goal', 'diet_goal']])
        
        # Optimized model parameters
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            encoded_features, df['selected_plan_id'],
            test_size=0.2,
            stratify=df['selected_plan_id']
        )
        
        model.fit(X_train, y_train)
        
        # Model evaluation
        print("\nModel Validation Report:")
        print(classification_report(y_val, model.predict(X_val)))
        
        # Save artifacts
        joblib.dump(model, 'model/nutrition_model.pkl')
        joblib.dump(encoder, 'model/feature_encoder.pkl')
        print("\n✅ Model successfully trained and saved")
        
    except Exception as e:
        print(f"\n❌ Model training failed: {str(e)}")
        raise

def get_rule_based_plan(user_data: dict) -> dict:
    """Advanced rule-based fallback system"""
    try:
        diet = user_data.get('diet', ['general'])[0]
        goal = user_data.get('goal', 'general-health')
        
        decision_matrix = {
            'diabetes-friendly': 17,
            'pregnancy': 16,
            'senior-health': 15,
            'high-protein': {
                'muscle-gain': 1,
                'mass-gain': 18,
                'default': 4
            },
            'low-carb': {
                'weight-loss': 5,
                'diabetes-friendly': 17,
                'default': 2
            },
            'budget-friendly': 19,
            'default': 0
        }
        
        if diet in decision_matrix and isinstance(decision_matrix[diet], int):
            return MEAL_PLANS[decision_matrix[diet]]
        
        if diet in decision_matrix and isinstance(decision_matrix[diet], dict):
            return MEAL_PLANS[decision_matrix[diet].get(goal, decision_matrix[diet]['default'])]
        
        return MEAL_PLANS[decision_matrix['default']]
    
    except Exception as e:
        print(f"Rule-based system error: {str(e)}")
        return MEAL_PLANS[0]

if __name__ == '__main__':
    print("🚀 Starting Nutrition Model Training...")
    try:
        train_model()
        print("✨ Training completed successfully")
    except Exception as e:
        print(f"🔥 Critical training error: {str(e)}")
