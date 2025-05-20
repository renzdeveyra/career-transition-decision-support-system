from flask import Flask, render_template, request, jsonify
from career_transition_system import CareerTransitionSystem
import json
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
system = CareerTransitionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get user profile from form
    user_profile = {
        "age": int(request.form.get('age')),
        "bpo_experience_years": float(request.form.get('bpo_experience_years')),
        "current_salary": int(request.form.get('current_salary')),
        "bpo_satisfaction": int(request.form.get('bpo_satisfaction')),
        "has_degree": request.form.get('has_degree') == 'true',
        "interests": request.form.getlist('interests'),
        "financial_pressure": request.form.get('financial_pressure'),
        "performance_level": request.form.get('performance_level')
    }
    
    # Run system
    results = system.validate_recommendation(user_profile)
    
    # Generate explanation
    explanation = system.explain_recommendation(results)
    
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

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Change the port from default 5000 to 8080
