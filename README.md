# Career Transition Decision Support System

A hybrid expert system and simulation model for providing career transition advice to BPO professionals in the Philippines.

## Overview

This system combines:

- A blackboard architecture-based expert system with multiple knowledge sources
- A simulation model to evaluate career decisions across different scenarios

## Installation

```bash
# Using Poetry (recommended)
poetry install

# Using pip
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Run with sample profile
python main.py

# Run with custom profile
python main.py --profile user_profiles/sample.json

# Save results to file
python main.py --output results.json

# Generate visualizations
python main.py --visualize
```

### API Usage

```python
from career_transition_system import CareerTransitionSystem

# Initialize system
system = CareerTransitionSystem()

# Define user profile
user_profile = {
    "age": 25,
    "bpo_experience_years": 3,
    "current_salary": 30000,
    "bpo_satisfaction": 6,
    "has_degree": True,
    "interests": ["technology", "leadership"],
    "financial_pressure": "medium"
}

# Get recommendation
recommendation = system.get_recommendation(user_profile)

# Simulate outcomes
simulation = system.simulate_outcomes(user_profile)

# Validate recommendation against simulation
validation = system.validate_recommendation(user_profile)
```

## Usage Examples

### Basic CLI Usage

```bash
# Run with default sample profile
python main.py

# Run in interactive mode
python main.py --interactive

# Process a specific profile and save results
python main.py --profile user_profiles/john_doe.json --output results/john_doe_results.json --visualize
```

### Batch Processing

Process multiple profiles at once:

```bash
python batch_process.py --input user_profiles/ --output results/
```

### Web Interface

Start the web application:

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

### API Integration

```python
from career_transition_system import CareerTransitionSystem

# Initialize system
system = CareerTransitionSystem()

# Define user profile
user_profile = {
    "age": 25,
    "bpo_experience_years": 3,
    "current_salary": 30000,
    "bpo_satisfaction": 6,
    "has_degree": True,
    "interests": ["technology", "leadership"],
    "financial_pressure": "medium",
    "performance_level": "good"
}

# Get comprehensive results
results = system.validate_recommendation(user_profile)

# Extract just the top recommendation
top_recommendation = results["expert_recommendation"]["top_recommendation"]["name"]
print(f"Top recommendation: {top_recommendation}")

# Get detailed explanation
explanation = system.explain_recommendation(results)
print(explanation)
```

## Architecture

### Expert System (Blackboard Architecture)

- **Blackboard**: Central data structure holding problem state
- **Knowledge Sources**: Domain experts contributing to solution
  - Career Counselor: Personality and interest alignment
  - Senior BPO Employee: Industry-specific insights
  - Academic Advisor: Educational pathway expertise
- **Control Shell**: Manages problem-solving process

### Simulation Model

Simulates career progression under different scenarios:

- Salary growth
- Job satisfaction
- Skill development
- Work-life balance

## Testing

Run the test suite:

```bash
python -m unittest discover tests
```

## License

MIT
