from flask import Flask, render_template, request, jsonify
from career_transition_system import CareerTransitionSystem
import json
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
# Initialize system with fixed seed for deterministic results
system = CareerTransitionSystem(seed=42)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get user profile from form
        user_profile = {
            # Personal information
            "age": int(request.form.get('age')),
            "has_degree": request.form.get('has_degree') == 'true',
            "highest_education": request.form.get('highest_education'),

            # Career information
            "bpo_experience_years": float(request.form.get('bpo_experience_years')),
            "current_role": request.form.get('current_role'),
            "current_salary": int(request.form.get('current_salary')),
            "bpo_satisfaction": int(request.form.get('bpo_satisfaction')),
            "performance_level": request.form.get('performance_level'),

            # Preferences
            "interests": request.form.getlist('interests'),
            "financial_pressure": request.form.get('financial_pressure'),
            "work_life_balance_importance": request.form.get('work_life_balance_importance'),

            # Alternative field information
            "identified_alternative_field": request.form.get('identified_alternative_field') == 'true',
            "alternative_field": request.form.get('alternative_field'),
            "researched_requirements": request.form.get('researched_requirements') == 'true',

            # Personality traits
            "personality_traits": {
                "conscientiousness": request.form.get('conscientiousness', 'medium'),
                "extroversion": request.form.get('extroversion', 'medium'),
                "openness": request.form.get('openness', 'medium')
            }
        }

        # Run system
        results = system.validate_recommendation(user_profile)

        # Generate explanation
        explanation = system.explain_recommendation(results)

        # Format explanation for web display (keep markdown format)
        # The client-side will convert markdown to HTML with proper styling

        # Convert visualization to base64 if it exists
        viz_data = None
        if results.get("visualization"):
            buf = io.BytesIO()
            results["visualization"].savefig(buf, format='png')
            buf.seek(0)
            viz_data = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({
            'explanation': explanation,
            'visualization': viz_data,
            'recommendation': results.get('expert_recommendation', {}).get('top_recommendation', {})
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Change the port from default 5000 to 8080

