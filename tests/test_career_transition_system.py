import unittest
from career_transition_system import CareerTransitionSystem, Blackboard

class TestCareerTransitionSystem(unittest.TestCase):
    def setUp(self):
        self.system = CareerTransitionSystem()
        self.sample_profile = {
            "age": 25,
            "bpo_experience_years": 3,
            "current_salary": 30000,
            "bpo_satisfaction": 6,
            "has_degree": True,
            "interests": ["technology", "leadership"]
        }
    
    def test_get_recommendation(self):
        result = self.system.get_recommendation(self.sample_profile)
        self.assertIsNotNone(result)
        self.assertIn("top_recommendation", result)
        
    def test_simulate_outcomes(self):
        results = self.system.simulate_outcomes(self.sample_profile)
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
        
    def test_validate_recommendation(self):
        results = self.system.validate_recommendation(self.sample_profile)
        self.assertIn("expert_recommendation", results)
        self.assertIn("validation", results)