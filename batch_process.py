from career_transition_system import CareerTransitionSystem
import json
import os
import pandas as pd
import argparse

def process_batch(input_dir, output_dir):
    """Process multiple user profiles in batch mode."""
    system = CareerTransitionSystem()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in input directory
    profile_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each profile
    results = []
    for profile_file in profile_files:
        profile_path = os.path.join(input_dir, profile_file)
        
        # Load profile
        with open(profile_path, 'r') as f:
            user_profile = json.load(f)
        
        # Process profile
        recommendation = system.validate_recommendation(user_profile)
        
        # Extract key information
        result = {
            'profile_name': profile_file,
            'age': user_profile.get('age'),
            'bpo_experience': user_profile.get('bpo_experience_years'),
            'top_recommendation': recommendation.get('expert_recommendation', {}).get('top_recommendation', {}).get('name', 'Unknown'),
            'confidence_score': recommendation.get('expert_recommendation', {}).get('top_recommendation', {}).get('confidence_score', 0)
        }
        
        results.append(result)
        
        # Save full results
        output_path = os.path.join(output_dir, f"result_{profile_file}")
        with open(output_path, 'w') as f:
            json.dump(recommendation, f, indent=2)
    
    # Create summary report
    df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, "summary_report.csv")
    df.to_csv(summary_path, index=False)
    
    return len(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process career profiles')
    parser.add_argument('--input', required=True, help='Directory containing profile JSON files')
    parser.add_argument('--output', required=True, help='Directory to save results')
    args = parser.parse_args()
    
    count = process_batch(args.input, args.output)
    print(f"Processed {count} profiles. Results saved to {args.output}")