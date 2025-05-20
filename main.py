from career_transition_system import CareerTransitionSystem
import argparse
import json

def load_profile(profile_path):
    """Load user profile from JSON file."""
    with open(profile_path, 'r') as f:
        return json.load(f)

def interactive_mode():
    """Run the system in interactive mode, collecting user input."""
    print("Career Transition Decision Support System - Interactive Mode")
    print("--------------------------------------------------------")
    
    # Collect personal information
    age = int(input("Age: "))
    has_degree = input("Do you have a college degree? (y/n): ").lower() == 'y'
    highest_education = input("Highest education level: ")
    
    # Collect career information
    bpo_experience_years = float(input("Years of BPO experience: "))
    current_role = input("Current role: ")
    current_salary = int(input("Current monthly salary (PHP): "))
    performance_level = input("Performance level (poor/average/good/excellent): ")
    
    # Collect preferences
    bpo_satisfaction = int(input("Satisfaction with BPO career (1-10): "))
    interests = input("Interests (comma-separated): ").split(',')
    interests = [i.strip() for i in interests]
    financial_pressure = input("Financial pressure (low/medium/high): ")
    
    # Create user profile
    user_profile = {
        "age": age,
        "has_degree": has_degree,
        "highest_education": highest_education,
        "bpo_experience_years": bpo_experience_years,
        "current_role": current_role,
        "current_salary": current_salary,
        "performance_level": performance_level,
        "bpo_satisfaction": bpo_satisfaction,
        "interests": interests,
        "financial_pressure": financial_pressure
    }
    
    return user_profile

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Career Transition Decision Support System')
    parser.add_argument('--profile', type=str, help='Path to user profile JSON file')
    parser.add_argument('--output', type=str, help='Path to save results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    # Create system
    system = CareerTransitionSystem()
    
    # Determine user profile source
    if args.interactive:
        user_profile = interactive_mode()
    elif args.profile:
        user_profile = load_profile(args.profile)
    else:
        # Use sample profile
        user_profile = {
            "age": 22,
            "bpo_experience_years": 2,
            "current_salary": 28000,
            "bpo_satisfaction": 8,
            "has_degree": True,
            "interests": ["technology", "leadership"],
            "financial_pressure": "medium",
            "performance_level": "good"
        }
    
    # Run system
    results = system.validate_recommendation(user_profile)
    
    # Generate explanation
    explanation = system.explain_recommendation(results)
    print(explanation)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Generate visualization if requested
    if args.visualize and results.get("visualization"):
        results["visualization"].savefig("career_path_comparison.png")
        print("Visualization saved as career_path_comparison.png")

if __name__ == "__main__":
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
  main()
  
