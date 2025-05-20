"""
Career Transition Decision Support System

This system implements a blackboard architecture-based expert system for career transition advice,
combined with a simulation model to evaluate career decisions across different scenarios.

The system focuses on providing guidance to young professionals in the Philippines,
particularly those in the BPO (Business Process Outsourcing) industry considering career transitions.

Classes:
    BlackboardSystem: Central system managing the blackboard and knowledge sources
    Blackboard: Central data structure holding problem state and partial solutions
    KnowledgeSource: Abstract base class for all knowledge sources
    CareerCounselor: Knowledge source representing a career counselor's expertise
    SeniorBPOEmployee: Knowledge source representing a senior BPO employee's expertise
    AcademicAdvisor: Knowledge source representing an academic advisor's expertise
    ControlShell: Controls the problem-solving process
    CareerSimulation: Simulates career progression outcomes under different scenarios
    CareerTransitionSystem: Main class integrating both expert system and simulation components

Usage:
    system = CareerTransitionSystem()
    recommendation = system.get_recommendation(user_profile)
    simulation = system.simulate_outcomes(user_profile)
    validation = system.validate_recommendation(user_profile)
"""

import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# --- PART 1: EXPERT SYSTEMS / BLACKBOARD ARCHITECTURE ---

class BlackboardSystem:
    """Central system managing the blackboard and knowledge sources."""

    def __init__(self):
        self.blackboard = Blackboard()
        self.knowledge_sources = []
        self.control_shell = ControlShell()

    def register_knowledge_source(self, knowledge_source):
        """Register a knowledge source with the blackboard system."""
        self.knowledge_sources.append(knowledge_source)

    def solve(self, user_profile):
        """Run the blackboard problem-solving process."""
        # Initialize blackboard with user profile
        self.blackboard.user_profile = user_profile

        # Initialize solution space
        self.blackboard.initialize_solution_space()

        iteration = 0
        max_iterations = 10  # Prevent infinite loops

        while not self.control_shell.is_solution_complete(self.blackboard) and iteration < max_iterations:
            # Select next knowledge source to run
            knowledge_source = self.control_shell.select_knowledge_source(
                self.knowledge_sources,
                self.blackboard
            )

            if knowledge_source is None:
                break

            # Apply selected knowledge source
            knowledge_source.contribute(self.blackboard)

            # Resolve conflicts if necessary
            self.control_shell.resolve_conflicts(self.blackboard)

            iteration += 1

        # Finalize solution
        return self.control_shell.finalize_solution(self.blackboard)


class Blackboard:
    """Central data structure holding problem state and partial solutions."""

    def __init__(self):
        # User profile information
        self.user_profile = {}

        # Partial solutions (career options being considered)
        self.career_options = []

        # Pros and cons for each option
        self.pros_cons = {}

        # Confidence scores for each option
        self.confidence_scores = {}

        # Recommendations from each expert
        self.expert_recommendations = {}

        # Conflicts identified
        self.conflicts = []

    def initialize_solution_space(self):
        """Initialize the solution space based on user profile."""
        # Extract basic information
        experience_years = self.user_profile.get('bpo_experience_years', 0)

        # Initialize some default career options based on BPO experience
        self.career_options = [
            {"id": "stay_bpo", "name": "Stay in BPO", "category": "bpo"},
            {"id": "advance_bpo", "name": "Advance in BPO (Team Lead/Specialist)", "category": "bpo"},
            {"id": "switch_tech", "name": "Switch to Tech/IT", "category": "transition"},
            {"id": "switch_business", "name": "Switch to Business Role", "category": "transition"},
            {"id": "switch_education", "name": "Switch to Education/Training", "category": "transition"},
            {"id": "switch_healthcare", "name": "Switch to Healthcare", "category": "transition"},
            {"id": "switch_creative", "name": "Switch to Creative/Design", "category": "transition"},
            {"id": "further_education", "name": "Pursue Further Education", "category": "education"}
        ]

        # Initialize pros/cons for each option with default values
        for option in self.career_options:
            option_id = option["id"]
            self.pros_cons[option_id] = {"pros": [], "cons": []}
            self.confidence_scores[option_id] = 0

            # Add default pros and cons based on option type
            if option_id == "stay_bpo":
                self.pros_cons[option_id]["pros"].append({
                    "factor": "stability",
                    "description": "Provides stable income and familiar work environment",
                    "weight": 0.7
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "growth_ceiling",
                    "description": "May have limited long-term growth potential",
                    "weight": 0.6
                })
            elif option_id == "advance_bpo":
                self.pros_cons[option_id]["pros"].append({
                    "factor": "career_progression",
                    "description": "Clear path for career advancement within the industry",
                    "weight": 0.75
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "responsibility_increase",
                    "description": "Increased responsibilities may affect work-life balance",
                    "weight": 0.5
                })
            elif option_id == "switch_tech":
                self.pros_cons[option_id]["pros"].append({
                    "factor": "industry_growth",
                    "description": "Tech industry offers numerous opportunities and growth",
                    "weight": 0.8
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "skill_gap",
                    "description": "May require significant upskilling or reskilling",
                    "weight": 0.65
                })
            elif option_id == "switch_business":
                self.pros_cons[option_id]["pros"].append({
                    "factor": "skill_transfer",
                    "description": "Many BPO skills transfer well to business roles",
                    "weight": 0.7
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "competition",
                    "description": "May face competition from business graduates",
                    "weight": 0.6
                })
            elif option_id == "switch_education":
                self.pros_cons[option_id]["pros"].append({
                    "factor": "work_life_balance",
                    "description": "Education sector often offers better work-life balance",
                    "weight": 0.85
                })
                self.pros_cons[option_id]["pros"].append({
                    "factor": "job_satisfaction",
                    "description": "Teaching and training roles often provide high job satisfaction",
                    "weight": 0.8
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "salary_adjustment",
                    "description": "May require initial salary adjustment compared to BPO",
                    "weight": 0.7
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "qualification_requirements",
                    "description": "May require specific teaching credentials or certifications",
                    "weight": 0.65
                })
            elif option_id == "switch_healthcare":
                self.pros_cons[option_id]["pros"].append({
                    "factor": "job_security",
                    "description": "Healthcare sector offers strong job security and demand",
                    "weight": 0.85
                })
                self.pros_cons[option_id]["pros"].append({
                    "factor": "meaningful_work",
                    "description": "Provides opportunity for meaningful, impactful work",
                    "weight": 0.8
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "training_requirements",
                    "description": "Requires specific healthcare training or certifications",
                    "weight": 0.75
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "high_stress",
                    "description": "Some healthcare roles can involve high stress levels",
                    "weight": 0.6
                })
            elif option_id == "switch_creative":
                self.pros_cons[option_id]["pros"].append({
                    "factor": "self_expression",
                    "description": "Allows for creativity and self-expression in work",
                    "weight": 0.85
                })
                self.pros_cons[option_id]["pros"].append({
                    "factor": "diverse_opportunities",
                    "description": "Creative fields offer diverse project opportunities",
                    "weight": 0.75
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "income_variability",
                    "description": "May involve variable income, especially when starting",
                    "weight": 0.8
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "competitive_field",
                    "description": "Creative industries can be highly competitive",
                    "weight": 0.7
                })
            elif option_id == "further_education":
                self.pros_cons[option_id]["pros"].append({
                    "factor": "qualification_boost",
                    "description": "Formal education can open more career opportunities",
                    "weight": 0.8
                })
                self.pros_cons[option_id]["cons"].append({
                    "factor": "time_commitment",
                    "description": "Requires significant time commitment alongside work",
                    "weight": 0.7
                })

        # Initialize empty expert recommendations
        self.expert_recommendations = {
            "career_counselor": None,
            "senior_bpo": None,
            "academic_advisor": None
        }


class KnowledgeSource:
    """Abstract base class for all knowledge sources."""

    def __init__(self, name):
        self.name = name

    def is_applicable(self, blackboard):
        """Determine if this knowledge source can contribute to the current blackboard state."""
        # Default implementation - override in subclasses
        return True

    def contribute(self, blackboard):
        """Contribute knowledge to the blackboard."""
        raise NotImplementedError("Subclasses must implement contribute()")


class CareerCounselor(KnowledgeSource):
    """Knowledge source representing a career counselor's expertise."""

    def __init__(self):
        super().__init__("Career Counselor")

    def is_applicable(self, blackboard):
        # A career counselor can always provide insights
        return True

    def contribute(self, blackboard):
        """Apply career counselor heuristics to the blackboard."""
        user = blackboard.user_profile

        # Apply personality-career fit assessment
        self._assess_personality_fit(blackboard)

        # Apply financial pressure heuristic
        if user.get('financial_pressure', 'medium') == 'high':
            # Add pros for financially stable options
            for option_id in ["stay_bpo", "advance_bpo"]:
                blackboard.pros_cons[option_id]["pros"].append({
                    "factor": "financial_stability",
                    "description": "Provides immediate financial stability for high-pressure situations",
                    "weight": 0.8
                })

            # Add cons for options with financial uncertainty
            for option_id in ["further_education", "switch_tech", "switch_business"]:
                blackboard.pros_cons[option_id]["cons"].append({
                    "factor": "financial_uncertainty",
                    "description": "May cause financial strain during transition period",
                    "weight": 0.7
                })

        # Apply passion-alignment heuristic
        interests = user.get('interests', [])

        if 'technology' in interests:
            blackboard.pros_cons["switch_tech"]["pros"].append({
                "factor": "passion_alignment",
                "description": "Aligns with expressed interest in technology",
                "weight": 0.9
            })
            blackboard.pros_cons["switch_tech"]["pros"].append({
                "factor": "growth_potential",
                "description": "Tech industry offers higher long-term salary growth potential",
                "weight": 0.8
            })
            blackboard.pros_cons["switch_tech"]["pros"].append({
                "factor": "future_proof",
                "description": "Tech skills are increasingly valuable across all industries",
                "weight": 0.85
            })
            blackboard.pros_cons["switch_tech"]["cons"].append({
                "factor": "initial_adjustment",
                "description": "May require initial salary adjustment during transition period",
                "weight": 0.6
            })

        if 'leadership' in interests:
            blackboard.pros_cons["advance_bpo"]["pros"].append({
                "factor": "passion_alignment",
                "description": "Aligns with expressed interest in leadership",
                "weight": 0.9
            })
            blackboard.pros_cons["advance_bpo"]["pros"].append({
                "factor": "experience_leverage",
                "description": "Leverages existing industry experience and connections",
                "weight": 0.85
            })
            blackboard.pros_cons["advance_bpo"]["pros"].append({
                "factor": "immediate_growth",
                "description": "Provides immediate career advancement without transition period",
                "weight": 0.8
            })
            blackboard.pros_cons["advance_bpo"]["cons"].append({
                "factor": "industry_limitation",
                "description": "Career growth may be limited to BPO industry",
                "weight": 0.7
            })

        if 'business' in interests or 'management' in interests:
            blackboard.pros_cons["switch_business"]["pros"].append({
                "factor": "passion_alignment",
                "description": "Aligns with expressed interest in business/management",
                "weight": 0.9
            })
            blackboard.pros_cons["switch_business"]["pros"].append({
                "factor": "transferable_skills",
                "description": "Customer service and communication skills transfer well to business roles",
                "weight": 0.8
            })
            blackboard.pros_cons["switch_business"]["cons"].append({
                "factor": "competition",
                "description": "Business roles often face high competition from candidates with business degrees",
                "weight": 0.7
            })

        if 'learning' in interests or 'academic' in interests:
            blackboard.pros_cons["further_education"]["pros"].append({
                "factor": "passion_alignment",
                "description": "Aligns with expressed interest in continued learning",
                "weight": 0.9
            })
            blackboard.pros_cons["further_education"]["pros"].append({
                "factor": "long_term_potential",
                "description": "Higher education can significantly increase long-term earning potential",
                "weight": 0.85
            })
            blackboard.pros_cons["further_education"]["cons"].append({
                "factor": "time_investment",
                "description": "Requires significant time investment alongside work responsibilities",
                "weight": 0.75
            })
            blackboard.pros_cons["further_education"]["cons"].append({
                "factor": "financial_investment",
                "description": "May require financial investment in tuition and materials",
                "weight": 0.7
            })

            # Also align with education career path
            blackboard.pros_cons["switch_education"]["pros"].append({
                "factor": "passion_alignment",
                "description": "Aligns with expressed interest in learning and academics",
                "weight": 0.9
            })

        if 'creative' in interests:
            blackboard.pros_cons["switch_creative"]["pros"].append({
                "factor": "passion_alignment",
                "description": "Aligns with expressed interest in creative work",
                "weight": 0.9
            })
            blackboard.pros_cons["switch_creative"]["pros"].append({
                "factor": "self_expression",
                "description": "Provides outlet for creative expression and innovation",
                "weight": 0.85
            })
            blackboard.pros_cons["switch_creative"]["cons"].append({
                "factor": "market_uncertainty",
                "description": "Creative markets can be unpredictable and competitive",
                "weight": 0.7
            })

        if 'specialized_bpo' in interests:
            blackboard.pros_cons["advance_bpo"]["pros"].append({
                "factor": "specialization_interest",
                "description": "Aligns with interest in specialized BPO roles",
                "weight": 0.85
            })

        # Make a recommendation
        self._make_recommendation(blackboard)

    def _assess_personality_fit(self, blackboard):
        """Assess personality fit with different career options."""
        user = blackboard.user_profile
        personality = user.get('personality_traits', {})

        # Check conscientiousness (detail-oriented, structured)
        if personality.get('conscientiousness', 'medium') == 'high':
            for option_id in ["stay_bpo", "advance_bpo"]:
                blackboard.pros_cons[option_id]["pros"].append({
                    "factor": "personality_fit",
                    "description": "Structured environment suits detail-oriented personality",
                    "weight": 0.6
                })

        # Check extroversion (social, outgoing)
        if personality.get('extroversion', 'medium') == 'high':
            for option_id in ["advance_bpo", "switch_business"]:
                blackboard.pros_cons[option_id]["pros"].append({
                    "factor": "personality_fit",
                    "description": "Role involves significant interpersonal interaction",
                    "weight": 0.5
                })

        # Check openness (creativity, new experiences)
        if personality.get('openness', 'medium') == 'high':
            for option_id in ["switch_tech", "switch_business", "further_education"]:
                blackboard.pros_cons[option_id]["pros"].append({
                    "factor": "personality_fit",
                    "description": "Opportunity for new experiences and creative problem-solving",
                    "weight": 0.7
                })

    def _make_recommendation(self, blackboard):
        """Formulate a recommendation based on counselor perspective."""
        user = blackboard.user_profile

        # Rule implementation from expert input
        # "IF the individual expresses significant dissatisfaction with the BPO environment OR
        # has identified a specific alternative field aligned with their interests and transferable skills
        # AND has researched the requirements and potential pathways in that field,
        # THEN recommend exploring entry-level roles or targeted upskilling opportunities in the alternative field,
        # WHILE advising them to maintain financial stability during the transition."

        dissatisfaction = user.get('bpo_satisfaction', 5) < 4  # On a scale of 1-10
        has_identified_alternative = user.get('identified_alternative_field', False)
        has_researched_requirements = user.get('researched_requirements', False)

        if dissatisfaction or (has_identified_alternative and has_researched_requirements):
            alternative_field = user.get('alternative_field')

            if alternative_field == 'tech':
                recommendation = "switch_tech"
            elif alternative_field == 'business':
                recommendation = "switch_business"
            elif alternative_field == 'education':
                recommendation = "switch_education"
            elif alternative_field == 'healthcare':
                recommendation = "switch_healthcare"
            elif alternative_field == 'creative':
                recommendation = "switch_creative"
            else:
                # If no specific field or default
                recommendation = "further_education"

            # Add stability caveat
            stability_note = "Maintain financial stability during transition by considering part-time options or gradual transition."

        else:
            # Without clear dissatisfaction or research, suggest advancing in current field
            recommendation = "advance_bpo"
            stability_note = "Leverage current skills while exploring other interests on the side."

        blackboard.expert_recommendations["career_counselor"] = {
            "recommendation": recommendation,
            "explanation": "Based on personality fit, expressed interests, and current satisfaction levels.",
            "additional_notes": stability_note
        }


class SeniorBPOEmployee(KnowledgeSource):
    """Knowledge source representing a senior BPO employee's expertise."""

    def __init__(self):
        super().__init__("Senior BPO Employee")

    def is_applicable(self, blackboard):
        # Senior BPO is especially relevant for those with BPO experience
        return blackboard.user_profile.get('bpo_experience_years', 0) > 0

    def contribute(self, blackboard):
        """Apply BPO industry-specific knowledge to the blackboard."""
        user = blackboard.user_profile
        experience_years = user.get('bpo_experience_years', 0)
        performance = user.get('performance_level', 'average')
        has_degree = user.get('has_degree', False)

        # Apply experience-based rules
        if experience_years >= 3:
            # Promotion potential in BPO
            blackboard.pros_cons["advance_bpo"]["pros"].append({
                "factor": "experience_advantage",
                "description": f"{experience_years} years of experience provides strong foundation for advancement",
                "weight": 0.8
            })

            # Transferable skills for business roles
            blackboard.pros_cons["switch_business"]["pros"].append({
                "factor": "transferable_skills",
                "description": "Existing communication and customer service skills transfer well",
                "weight": 0.7
            })
        else:
            # Limited experience
            blackboard.pros_cons["advance_bpo"]["cons"].append({
                "factor": "experience_limitation",
                "description": "May need more experience before promotion opportunities",
                "weight": 0.6
            })

        # Apply performance-based insights
        if performance == 'excellent':
            blackboard.pros_cons["advance_bpo"]["pros"].append({
                "factor": "performance_record",
                "description": "Excellent performance record indicates high potential for advancement",
                "weight": 0.9
            })

        # Apply education-based insights
        if not has_degree:
            # Challenges for non-degree holders
            blackboard.pros_cons["advance_bpo"]["cons"].append({
                "factor": "education_limitation",
                "description": "Some higher management roles may require degrees",
                "weight": 0.5
            })

            blackboard.pros_cons["switch_business"]["cons"].append({
                "factor": "education_requirement",
                "description": "Many business roles may require degrees for entry",
                "weight": 0.7
            })

            blackboard.pros_cons["further_education"]["pros"].append({
                "factor": "career_expansion",
                "description": "Would remove education barrier to career advancement",
                "weight": 0.8
            })

        # Make a recommendation
        self._make_recommendation(blackboard)

    def _make_recommendation(self, blackboard):
        """Formulate a recommendation based on BPO industry perspective."""
        user = blackboard.user_profile

        # Rule implementation from expert input
        # "IF the individual has consistently met or exceeded performance targets in their BPO role
        # AND expresses interest in leadership or specialized functions within BPO,
        # THEN recommend pursuing internal promotion opportunities or seeking roles in departments
        # like Training or Quality Assurance.
        # ELSE IF they express strong dissatisfaction with BPO AND have identified a specific alternative field,
        # THEN recommend researching that field's requirements and focusing on acquiring necessary skills,
        # perhaps through online courses, while continuing to work in BPO to maintain income."

        good_performance = user.get('performance_level') in ['good', 'excellent']
        interest_in_bpo_advancement = 'leadership' in user.get('interests', []) or 'specialized_bpo' in user.get('interests', [])

        if good_performance and interest_in_bpo_advancement:
            recommendation = "advance_bpo"
            explanation = "Strong performance and interest in leadership/specialized roles indicate promotion potential."

        elif user.get('bpo_satisfaction', 5) < 3 and user.get('identified_alternative_field', False):
            alternative_field = user.get('alternative_field')

            if alternative_field == 'tech':
                recommendation = "switch_tech"
            elif alternative_field == 'business':
                recommendation = "switch_business"
            elif alternative_field == 'education':
                recommendation = "switch_education"
            elif alternative_field == 'healthcare':
                recommendation = "switch_healthcare"
            elif alternative_field == 'creative':
                recommendation = "switch_creative"
            else:
                recommendation = "further_education"

            explanation = "Given dissatisfaction and interest in a specific alternative, transition is advisable while maintaining BPO income."

        else:
            # Default to staying but improving position
            recommendation = "stay_bpo"
            explanation = "Current BPO position provides stability while exploring options and building skills."

        blackboard.expert_recommendations["senior_bpo"] = {
            "recommendation": recommendation,
            "explanation": explanation,
            "additional_notes": "Consider specialized roles within BPO like Training or Quality Assurance as alternatives to traditional agent path."
        }


class AcademicAdvisor(KnowledgeSource):
    """Knowledge source representing an academic advisor's expertise."""

    def __init__(self):
        super().__init__("Academic Advisor")

    def is_applicable(self, blackboard):
        # Academic advisor can always provide educational insights
        return True

    def contribute(self, blackboard):
        """Apply educational pathway knowledge to the blackboard."""
        user = blackboard.user_profile

        has_degree = user.get('has_degree', False)
        interests = user.get('interests', [])
        financial_pressure = user.get('financial_pressure', 'medium')

        # Assess educational requirements for paths
        if not has_degree:
            # Educational recommendations for non-degree holders
            blackboard.pros_cons["further_education"]["pros"].append({
                "factor": "qualification_gap",
                "description": "Formal education would address qualification gap for many roles",
                "weight": 0.8
            })

            if financial_pressure == 'high':
                blackboard.pros_cons["further_education"]["cons"].append({
                    "factor": "financial_strain",
                    "description": "Full-time education may cause financial strain",
                    "weight": 0.9
                })

                # Alternative certifications
                for option_id in ["switch_tech", "switch_business"]:
                    blackboard.pros_cons[option_id]["pros"].append({
                        "factor": "targeted_certifications",
                        "description": "Shorter certification paths available as alternatives to full degrees",
                        "weight": 0.7
                    })

        # Field-specific educational insights
        if 'technology' in interests:
            blackboard.pros_cons["switch_tech"]["pros"].append({
                "factor": "learning_pathway",
                "description": "Many tech roles accessible through bootcamps or focused certifications",
                "weight": 0.8
            })

        if 'business' in interests:
            if has_degree:
                blackboard.pros_cons["switch_business"]["pros"].append({
                    "factor": "credential_advantage",
                    "description": "Existing degree provides foundation for business role transition",
                    "weight": 0.7
                })
            else:
                blackboard.pros_cons["switch_business"]["cons"].append({
                    "factor": "credential_gap",
                    "description": "Business roles often require formal education credentials",
                    "weight": 0.6
                })

        # Make a recommendation
        self._make_recommendation(blackboard)

    def _make_recommendation(self, blackboard):
        """Formulate a recommendation based on academic perspective."""
        user = blackboard.user_profile

        # Rule implementation from expert input
        # "IF the individual expresses a clear interest in a specific alternative field AND
        # that field requires specific technical skills not typically gained in BPO AND
        # there are accessible short-term educational programs (certificates, bootcamps, community college courses)
        # that teach those skills with a strong track record of graduate employment in that field,
        # THEN recommend enrolling in such a program WHILE advising them to seek part-time work or
        # continue in BPO part-time if financially necessary."

        has_clear_interest = user.get('identified_alternative_field', False)
        alternative_field = user.get('alternative_field', None)
        financial_pressure = user.get('financial_pressure', 'medium')

        if has_clear_interest:
            if alternative_field == 'tech':
                recommendation = "switch_tech"
                notes = "Consider bootcamps or IT certifications while maintaining part-time BPO work."

            elif alternative_field == 'business':
                if user.get('has_degree', False):
                    recommendation = "switch_business"
                    notes = "Leverage existing degree and pursue targeted business certifications."
                else:
                    recommendation = "further_education"
                    notes = "Consider business administration courses at community college level."

            elif alternative_field == 'education':
                recommendation = "switch_education"
                notes = "Consider education certifications or teaching credentials while maintaining part-time work."

            elif alternative_field == 'healthcare':
                recommendation = "switch_healthcare"
                notes = "Research healthcare certifications that align with your interests and time constraints."

            elif alternative_field == 'creative':
                recommendation = "switch_creative"
                notes = "Build a portfolio through freelance work while maintaining your current position for stability."

            else:
                recommendation = "further_education"
                notes = "A structured educational program would help clarify your path forward."

        elif not user.get('has_degree', False) and financial_pressure != 'high':
            recommendation = "further_education"
            notes = "A degree would open more advancement opportunities both in and outside BPO."

        else:
            # Default to skill development while staying in current role
            recommendation = "stay_bpo"
            notes = "Consider online learning and certifications while maintaining current position."

        blackboard.expert_recommendations["academic_advisor"] = {
            "recommendation": recommendation,
            "explanation": "Based on educational requirements and learning pathways available.",
            "additional_notes": notes
        }


class ControlShell:
    """Manages the problem-solving process and resolves conflicts between experts."""

    def select_knowledge_source(self, knowledge_sources, blackboard):
        """Select the next knowledge source to apply."""
        # Filter to applicable knowledge sources
        applicable_sources = [ks for ks in knowledge_sources if ks.is_applicable(blackboard)]

        # Check if there are any applicable sources left
        if not applicable_sources:
            return None

        # For simplicity, return the first applicable source that hasn't contributed yet
        for source in applicable_sources:
            if source.name not in blackboard.expert_recommendations or blackboard.expert_recommendations[source.name] is None:
                return source

        # All sources have contributed
        return None

    def is_solution_complete(self, blackboard):
        """Determine if the solution is complete."""
        # Solution is complete if all experts have contributed
        expected_experts = ["Career Counselor", "Senior BPO Employee", "Academic Advisor"]

        for expert in expected_experts:
            expert_key = expert.lower().replace(" ", "_")
            if expert_key not in blackboard.expert_recommendations or blackboard.expert_recommendations[expert_key] is None:
                return False

        return True

    def resolve_conflicts(self, blackboard):
        """Resolve conflicts between expert recommendations."""
        # Collect all recommendations
        recommendations = {
            expert: details["recommendation"]
            for expert, details in blackboard.expert_recommendations.items()
            if details is not None
        }

        # If we have multiple experts recommending different things, note the conflict
        if len(set(recommendations.values())) > 1:
            conflicting_experts = []
            for expert, rec in recommendations.items():
                conflicting_experts.append({
                    "expert": expert,
                    "recommendation": rec
                })

            if conflicting_experts and len(conflicting_experts) > 1:
                blackboard.conflicts.append({
                    "type": "recommendation_conflict",
                    "experts": conflicting_experts
                })

    def finalize_solution(self, blackboard):
        """Produce final recommendations based on all expert input."""
        # Calculate confidence scores based on pros/cons
        self._calculate_confidence_scores(blackboard)

        # Weight the expert recommendations
        final_scores = self._weight_expert_recommendations(blackboard)

        # Create the final solution
        solution = {
            "career_options": blackboard.career_options,
            "option_details": {},
            "top_recommendation": None,
            "confidence_score": 0,
            "alternative_recommendation": None,
            "considerations": [],
            "expert_recommendations": blackboard.expert_recommendations,
            "conflicts": blackboard.conflicts
        }

        # Add details for each option
        for option in blackboard.career_options:
            option_id = option["id"]
            solution["option_details"][option_id] = {
                "name": option["name"],
                "category": option["category"],
                "pros": blackboard.pros_cons[option_id]["pros"],
                "cons": blackboard.pros_cons[option_id]["cons"],
                "confidence_score": blackboard.confidence_scores[option_id],
                "final_score": final_scores.get(option_id, 0)
            }

        # Determine top recommendations
        if final_scores:
            # Sort options by final score
            sorted_options = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

            # Top recommendation
            top_id, top_score = sorted_options[0]
            top_option = next((o for o in blackboard.career_options if o["id"] == top_id), None)

            solution["top_recommendation"] = {
                "id": top_id,
                "name": top_option["name"] if top_option else top_id,
                "score": top_score
            }

            # Alternative recommendation (if available)
            if len(sorted_options) > 1:
                alt_id, alt_score = sorted_options[1]
                alt_option = next((o for o in blackboard.career_options if o["id"] == alt_id), None)

                solution["alternative_recommendation"] = {
                    "id": alt_id,
                    "name": alt_option["name"] if alt_option else alt_id,
                    "score": alt_score
                }

        # Add considerations based on user profile
        user = blackboard.user_profile

        if user.get('financial_pressure', 'medium') == 'high':
            solution["considerations"].append({
                "factor": "financial_stability",
                "description": "Financial stability is a primary concern. Consider options that maintain steady income."
            })

        if not user.get('has_degree', False):
            solution["considerations"].append({
                "factor": "education_credentials",
                "description": "Consider how educational credentials might impact long-term career ceiling."
            })

        return solution

    def _calculate_confidence_scores(self, blackboard):
        """Calculate confidence scores for each option based on pros/cons."""
        for option_id in blackboard.confidence_scores:
            pros = blackboard.pros_cons[option_id]["pros"]
            cons = blackboard.pros_cons[option_id]["cons"]

            # Calculate weighted score from pros and cons
            pro_score = sum(p.get("weight", 0.5) for p in pros)
            con_score = sum(c.get("weight", 0.5) for c in cons)

            # Confidence calculation
            # Scale to 0-100 and adjust based on number of factors
            pro_factor = pro_score * (10 / (1 + len(pros))) if pros else 0
            con_factor = con_score * (10 / (1 + len(cons))) if cons else 0

            confidence = pro_factor - (con_factor * 0.8)  # Cons weighted slightly less

            # Scale to 0-100 range and ensure non-negative
            scaled_confidence = max(0, min(100, confidence * 10))

            blackboard.confidence_scores[option_id] = scaled_confidence

    def _weight_expert_recommendations(self, blackboard):
        """Apply weights to expert recommendations."""
        final_scores = {}

        # Initialize with confidence scores
        for option_id, score in blackboard.confidence_scores.items():
            final_scores[option_id] = score

        # Expert recommendation weights
        expert_weights = {
            "career_counselor": 0.35,  # Weighted for personality-career fit
            "senior_bpo": 0.35,        # Weighted for industry-specific knowledge
            "academic_advisor": 0.30   # Weighted for educational pathways
        }

        # Expert recommendation boost
        expert_boost = 30  # Points to add for a recommendation

        # Apply expert boosts
        for expert, details in blackboard.expert_recommendations.items():
            if details is not None and details["recommendation"] in final_scores:
                weight = expert_weights.get(expert, 0.33)
                final_scores[details["recommendation"]] += expert_boost * weight

        return final_scores


# --- PART 2: MODELING AND SIMULATION ---

class CareerSimulation:
    """Simulates career progression outcomes under different scenarios."""

    def __init__(self, seed=42):
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Career paths and their properties
        self.career_paths = {
            "stay_bpo": {
                "salary_growth_rate": 0.05,  # 5% annual growth
                "promotion_probability": 0.15,  # 15% chance per year
                "job_satisfaction_baseline": 6.0,  # Scale 1-10
                "job_satisfaction_variance": 1.0,
                "stability": 0.8  # High stability
            },
            "advance_bpo": {
                "salary_growth_rate": 0.08,  # 8% annual growth
                "promotion_probability": 0.25,  # 25% chance per year
                "job_satisfaction_baseline": 7.0,
                "job_satisfaction_variance": 1.5,
                "stability": 0.75
            },
            "switch_tech": {
                "salary_growth_rate": 0.10,  # 10% annual growth
                "promotion_probability": 0.20,
                "job_satisfaction_baseline": 7.5,
                "job_satisfaction_variance": 2.0,
                "stability": 0.7
            },
            "switch_business": {
                "salary_growth_rate": 0.07,
                "promotion_probability": 0.18,
                "job_satisfaction_baseline": 7.0,
                "job_satisfaction_variance": 1.8,
                "stability": 0.65
            },
            "switch_education": {
                "salary_growth_rate": 0.06,  # Moderate growth
                "promotion_probability": 0.15,
                "job_satisfaction_baseline": 8.5,  # High satisfaction
                "job_satisfaction_variance": 1.2,
                "stability": 0.85  # High stability
            },
            "switch_healthcare": {
                "salary_growth_rate": 0.08,  # Good growth
                "promotion_probability": 0.18,
                "job_satisfaction_baseline": 7.8,
                "job_satisfaction_variance": 1.5,
                "stability": 0.80  # High stability
            },
            "switch_creative": {
                "salary_growth_rate": 0.07,  # Variable growth
                "promotion_probability": 0.16,
                "job_satisfaction_baseline": 8.2,  # High satisfaction potential
                "job_satisfaction_variance": 2.5,  # But high variance
                "stability": 0.60  # Lower stability
            },
            "further_education": {
                "salary_growth_rate": 0.12,  # Higher long-term growth
                "promotion_probability": 0.22,
                "job_satisfaction_baseline": 8.0,
                "job_satisfaction_variance": 1.5,
                "stability": 0.6  # Lower initial stability due to transition
            }
        }

        # Simulation parameters
        self.time_horizon = 5  # Simulate 5 years
        self.monte_carlo_iterations = 100  # Number of simulations to run

    def simulate_career_path(self, path_id, user_profile, scenario_type="average"):
        """Simulate outcomes for a specific career path."""
        # Get path properties
        path = self.career_paths.get(path_id)
        if not path:
            return None

        # Adjust parameters based on user profile
        path = self._adjust_path_for_user(path, user_profile)

        # Adjust parameters based on scenario type
        path = self._adjust_path_for_scenario(path, scenario_type)

        # Initialize metrics
        initial_salary = user_profile.get('current_salary', 25000)  # Default in PHP
        current_salary = initial_salary

        # Apply initial salary adjustments based on career path
        if path_id in ["switch_tech", "switch_business"]:
            # Initial salary adjustment for switching to tech or business
            current_salary = initial_salary * 0.9  # 10% reduction initially

        elif path_id == "switch_education":
            # Initial salary adjustment for switching to education
            current_salary = initial_salary * 0.85  # 15% reduction initially

        elif path_id == "switch_healthcare":
            # Initial salary adjustment for switching to healthcare
            current_salary = initial_salary * 0.8  # 20% reduction initially (but higher long-term potential)

        elif path_id == "switch_creative":
            # Initial salary adjustment for switching to creative fields
            current_salary = initial_salary * 0.75  # 25% reduction initially (more significant adjustment)

        elif path_id == "further_education":
            # Further reduction for education period
            current_salary = initial_salary * 0.6  # 40% reduction during education

        # Simulation results
        years = []
        salaries = []
        satisfaction = []
        stability = []

        # Run simulation for time horizon
        for year in range(self.time_horizon + 1):  # +1 to include initial year
            # Record current state
            years.append(year)
            salaries.append(current_salary)

            # Calculate satisfaction (baseline with some randomness)
            current_satisfaction = min(10, max(1,
                np.random.normal(
                    path["job_satisfaction_baseline"],
                    path["job_satisfaction_variance"] / 2
                )
            ))
            satisfaction.append(current_satisfaction)

            # Calculate stability
            current_stability = path["stability"] * (1 + 0.05 * year)  # Stability increases with time in role
            current_stability = min(0.99, current_stability)  # Cap at 99%
            stability.append(current_stability)

            # Update for next year if not at the end
            if year < self.time_horizon:
                # Apply salary growth
                current_salary *= (1 + path["salary_growth_rate"])

                # Apply promotion chance
                if np.random.random() < path["promotion_probability"]:
                    current_salary *= 1.15  # 15% boost for promotion

        # Calculate aggregate metrics
        final_salary = salaries[-1]
        salary_growth = (final_salary / initial_salary - 1) * 100  # Percentage
        avg_satisfaction = sum(satisfaction) / len(satisfaction)
        avg_stability = sum(stability) / len(stability)

        # Return simulation results
        return {
            "path_id": path_id,
            "scenario": scenario_type,
            "years": years,
            "salaries": salaries,
            "satisfaction": satisfaction,
            "stability": stability,
            "final_salary": final_salary,
            "salary_growth_pct": salary_growth,
            "avg_satisfaction": avg_satisfaction,
            "avg_stability": avg_stability
        }

    def simulate_all_paths(self, user_profile, scenario_type="average"):
        """Simulate all available career paths for comparison."""
        results = {}

        for path_id in self.career_paths:
            results[path_id] = self.simulate_career_path(path_id, user_profile, scenario_type)

        return results

    def run_monte_carlo(self, path_id, user_profile):
        """Run Monte Carlo simulation to account for uncertainty."""
        results = {
            "best_case": [],
            "average_case": [],
            "worst_case": []
        }

        # Run simulations for each scenario
        for scenario in ["best_case", "average_case", "worst_case"]:
            for _ in range(self.monte_carlo_iterations):
                sim_result = self.simulate_career_path(path_id, user_profile, scenario)
                if sim_result:
                    results[scenario].append(sim_result)

        # Calculate averages and confidence intervals
        summary = {}
        for scenario, sims in results.items():
            if sims:
                # Extract metrics of interest
                final_salaries = [sim["final_salary"] for sim in sims]
                salary_growth = [sim["salary_growth_pct"] for sim in sims]
                satisfaction = [sim["avg_satisfaction"] for sim in sims]
                stability = [sim["avg_stability"] for sim in sims]

                # Calculate statistics
                summary[scenario] = {
                    "final_salary": {
                        "mean": np.mean(final_salaries),
                        "median": np.median(final_salaries),
                        "std_dev": np.std(final_salaries),
                        "percentile_25": np.percentile(final_salaries, 25),
                        "percentile_75": np.percentile(final_salaries, 75)
                    },
                    "salary_growth_pct": {
                        "mean": np.mean(salary_growth),
                        "median": np.median(salary_growth),
                        "std_dev": np.std(salary_growth)
                    },
                    "satisfaction": {
                        "mean": np.mean(satisfaction),
                        "median": np.median(satisfaction),
                        "std_dev": np.std(satisfaction)
                    },
                    "stability": {
                        "mean": np.mean(stability),
                        "median": np.median(stability),
                        "std_dev": np.std(stability)
                    }
                }

        return summary

    def _adjust_path_for_user(self, path, user_profile):
        """Adjust path parameters based on user profile."""
        adjusted_path = path.copy()

        # Adjust based on education
        if user_profile.get('has_degree', False):
            adjusted_path["promotion_probability"] *= 1.2  # 20% higher promotion chance
            adjusted_path["salary_growth_rate"] *= 1.1  # 10% higher salary growth

        # Adjust based on performance
        performance = user_profile.get('performance_level', 'average')
        if performance == 'excellent':
            adjusted_path["promotion_probability"] *= 1.3
            adjusted_path["salary_growth_rate"] *= 1.2
        elif performance == 'good':
            adjusted_path["promotion_probability"] *= 1.1
            adjusted_path["salary_growth_rate"] *= 1.1
        elif performance == 'poor':
            adjusted_path["promotion_probability"] *= 0.8
            adjusted_path["salary_growth_rate"] *= 0.9

        # Adjust based on experience
        experience = user_profile.get('bpo_experience_years', 0)
        if experience >= 5:
            adjusted_path["promotion_probability"] *= 1.2
        elif experience >= 3:
            adjusted_path["promotion_probability"] *= 1.1

        return adjusted_path

    def _adjust_path_for_scenario(self, path, scenario_type):
        """Adjust path parameters based on scenario type."""
        adjusted_path = path.copy()

        if scenario_type == "best_case":
            adjusted_path["salary_growth_rate"] *= 1.3
            adjusted_path["promotion_probability"] *= 1.5
            adjusted_path["job_satisfaction_baseline"] = min(10, adjusted_path["job_satisfaction_baseline"] * 1.2)
            adjusted_path["stability"] = min(0.95, adjusted_path["stability"] * 1.2)

        elif scenario_type == "worst_case":
            adjusted_path["salary_growth_rate"] *= 0.7
            adjusted_path["promotion_probability"] *= 0.5
            adjusted_path["job_satisfaction_baseline"] *= 0.8
            adjusted_path["stability"] *= 0.8

        return adjusted_path

    def validate_against_expert_output(self, simulation_results, expert_system_output):
        """Compare simulation results against expert system recommendations."""
        validation_results = {
            "alignment": {},
            "conflicts": [],
            "overall_alignment": 0.0
        }

        # Get top recommendation from expert system
        expert_top_rec = expert_system_output.get("top_recommendation", {}).get("id")

        if not expert_top_rec or expert_top_rec not in simulation_results:
            return {"error": "Unable to validate: Missing expert recommendation or simulation data"}

        # Compare performance metrics for top recommendation vs alternatives
        top_rec_results = simulation_results.get(expert_top_rec, {}).get("average_case", {})

        # Check alignment for each alternative
        alignments = []

        for path_id, results in simulation_results.items():
            if path_id == expert_top_rec:
                continue

            path_results = results.get("average_case", {})

            # Compare key metrics
            metrics_comparison = {}
            conflict_found = False

            # Compare salary growth
            if "salary_growth_pct" in top_rec_results and "salary_growth_pct" in path_results:
                top_growth = top_rec_results["salary_growth_pct"]  # Now treating as float
                alt_growth = path_results["salary_growth_pct"]     # Now treating as float

                metrics_comparison["salary_growth"] = {
                    "expert_rec": top_growth,
                    "alternative": alt_growth,
                    "difference": top_growth - alt_growth,
                    "expert_better": top_growth > alt_growth
                }

                if alt_growth > top_growth + 15:  # If alternative is >15% better
                    conflict_found = True

            # Compare satisfaction
            if "avg_satisfaction" in top_rec_results and "avg_satisfaction" in path_results:
                top_sat = top_rec_results["avg_satisfaction"]  # Use average satisfaction (float)
                alt_sat = path_results["avg_satisfaction"]     # Use average satisfaction (float)

                metrics_comparison["satisfaction"] = {
                    "expert_rec": top_sat,
                    "alternative": alt_sat,
                    "difference": top_sat - alt_sat,
                    "expert_better": top_sat > alt_sat
                }

                if alt_sat > top_sat + 1.5:  # If alternative is >1.5 points better (on 1-10 scale)
                    conflict_found = True

            # Compare stability
            if "avg_stability" in top_rec_results and "avg_stability" in path_results:
                top_stab = top_rec_results["avg_stability"]  # Use average stability (float)
                alt_stab = path_results["avg_stability"]     # Use average stability (float)

                metrics_comparison["stability"] = {
                    "expert_rec": top_stab,
                    "alternative": alt_stab,
                    "difference": top_stab - alt_stab,
                    "expert_better": top_stab > alt_stab
                }

                if alt_stab > top_stab + 0.2:  # If alternative is >0.2 better (on 0-1 scale)
                    conflict_found = True

            # Record alignment result
            path_alignment = {
                "path_id": path_id,
                "metrics_comparison": metrics_comparison,
                "conflict_detected": conflict_found
            }

            alignments.append(path_alignment)

            # Record conflicts
            if conflict_found:
                validation_results["conflicts"].append({
                    "expert_recommendation": expert_top_rec,
                    "simulation_preferred": path_id,
                    "metrics": metrics_comparison
                })

        # Calculate overall alignment score (higher is better)
        num_alignments = len(alignments)
        num_conflicts = len(validation_results["conflicts"])

        if num_alignments > 0:
            alignment_score = (num_alignments - num_conflicts) / num_alignments
            validation_results["overall_alignment"] = max(0, alignment_score)

        # Add alignment details
        validation_results["alignment"] = {a["path_id"]: a for a in alignments}

        return validation_results

    def visualize_comparison(self, simulation_results, recommended_path):
        """Create visualizations comparing different career paths."""
        # Extract time series data
        paths = list(simulation_results.keys())

        # Set up figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Career Path Comparison', fontsize=16)

        # Colors with recommended path highlighted
        colors = {}
        for path in paths:
            if path == recommended_path:
                colors[path] = 'red'  # Highlight recommended path
            else:
                colors[path] = None  # Default color

        # Plot 1: Salary progression
        ax1 = axes[0, 0]
        for path in paths:
            result = simulation_results[path]
            ax1.plot(result["years"], result["salaries"], label=path, color=colors[path],
                     linewidth=3 if path == recommended_path else 2)
        ax1.set_title('Salary Progression')
        ax1.set_xlabel('Years')
        ax1.set_ylabel('Salary (PHP)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Job Satisfaction
        ax2 = axes[0, 1]
        for path in paths:
            result = simulation_results[path]
            ax2.plot(result["years"], result["satisfaction"], label=path, color=colors[path],
                     linewidth=3 if path == recommended_path else 2)
        ax2.set_title('Job Satisfaction')
        ax2.set_xlabel('Years')
        ax2.set_ylabel('Satisfaction (1-10)')
        ax2.set_ylim(0, 10)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Plot 3: Job Stability
        ax3 = axes[1, 0]
        for path in paths:
            result = simulation_results[path]
            ax3.plot(result["years"], result["stability"], label=path, color=colors[path],
                     linewidth=3 if path == recommended_path else 2)
        ax3.set_title('Job Stability')
        ax3.set_xlabel('Years')
        ax3.set_ylabel('Stability (0-1)')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Plot 4: Final Comparison
        ax4 = axes[1, 1]

        # Extract final values for comparison
        labels = []
        salary_growth = []
        satisfaction = []
        stability = []

        for path in paths:
            labels.append(path)
            result = simulation_results[path]
            salary_growth.append(result["salary_growth_pct"])
            satisfaction.append(result["avg_satisfaction"])
            stability.append(result["avg_stability"])

        # Width of bars
        width = 0.25

        # Positions for bars
        x = np.arange(len(labels))

        # Create bars
        ax4.bar(x - width, salary_growth, width, label='Salary Growth %')
        ax4.bar(x, satisfaction, width, label='Avg Satisfaction')
        ax4.bar(x + width, [s * 10 for s in stability], width, label='Stability (x10)')

        # Add labels and legend
        ax4.set_title('Final Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45)
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Return figure for displaying or saving
        return fig


# --- PART 3: INTEGRATION AND USAGE ---

class CareerTransitionSystem:
    """Main class integrating both expert system and simulation components."""

    def __init__(self, seed=42):
        # Initialize expert system components
        self.blackboard_system = BlackboardSystem()

        # Register knowledge sources
        self.blackboard_system.register_knowledge_source(CareerCounselor())
        self.blackboard_system.register_knowledge_source(SeniorBPOEmployee())
        self.blackboard_system.register_knowledge_source(AcademicAdvisor())

        # Initialize simulation component with fixed seed for reproducibility
        self.simulation = CareerSimulation(seed=seed)

    def get_recommendation(self, user_profile):
        """Get career recommendation based on user profile."""
        # Run expert system
        expert_recommendation = self.blackboard_system.solve(user_profile)

        # Return recommendation
        return expert_recommendation

    def simulate_outcomes(self, user_profile, recommended_path_id=None):
        """Simulate outcomes for all paths or a recommended path."""
        # Simulate all paths with basic setup
        simulation_results = {}

        for path_id in self.simulation.career_paths:
            simulation_results[path_id] = self.simulation.simulate_career_path(path_id, user_profile)

        # If a specific recommended path is provided, run Monte Carlo for it
        if recommended_path_id and recommended_path_id in self.simulation.career_paths:
            monte_carlo_results = self.simulation.run_monte_carlo(recommended_path_id, user_profile)

            # Add Monte Carlo results to the main results dictionary
            simulation_results[recommended_path_id + "_monte_carlo"] = monte_carlo_results

        return simulation_results

    def validate_recommendation(self, user_profile):
        """Run both systems and validate their alignment."""
        # Get expert recommendation
        expert_recommendation = self.get_recommendation(user_profile)

        # Get simulation results
        simulation_results = {}
        for path_id in self.simulation.career_paths:
            # Run for all scenario types
            path_results = {
                "best_case": self.simulation.simulate_career_path(path_id, user_profile, "best_case"),
                "average_case": self.simulation.simulate_career_path(path_id, user_profile, "average_case"),
                "worst_case": self.simulation.simulate_career_path(path_id, user_profile, "worst_case")
            }
            simulation_results[path_id] = path_results

        # Validate alignment
        validation_results = self.simulation.validate_against_expert_output(
            simulation_results, expert_recommendation
        )

        # Create visualization
        if expert_recommendation.get("top_recommendation", {}).get("id"):
            recommended_path = expert_recommendation["top_recommendation"]["id"]

            # Extract average case results for visualization
            vis_results = {}
            for path_id, scenarios in simulation_results.items():
                vis_results[path_id] = scenarios["average_case"]

            visualization = self.simulation.visualize_comparison(vis_results, recommended_path)
        else:
            visualization = None

        # Return comprehensive results
        return {
            "expert_recommendation": expert_recommendation,
            "simulation_results": simulation_results,
            "validation": validation_results,
            "visualization": visualization
        }

    def explain_recommendation(self, results):
        """Generate a human-readable explanation of the recommendation."""
        explanation = []

        # Get top recommendation
        top_rec = results.get("expert_recommendation", {}).get("top_recommendation", {})
        if not top_rec:
            return "Unable to generate recommendation with the provided information."

        top_path_id = top_rec.get("id")
        top_path_name = top_rec.get("name")

        # Get confidence score - first try to get from option_details, then from confidence_scores
        option_details = results.get("expert_recommendation", {}).get("option_details", {})
        if top_path_id in option_details and "confidence_score" in option_details[top_path_id]:
            confidence_score = option_details[top_path_id]["confidence_score"]
        else:
            confidence_score = results.get("expert_recommendation", {}).get("confidence_scores", {}).get(top_path_id, 0)

        # Get alternative recommendation
        alt_rec = results.get("expert_recommendation", {}).get("alternative_recommendation", {})
        alt_path_name = alt_rec.get("name") if alt_rec else None

        # Format confidence level
        confidence_level = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 40 else "Low"

        # Build explanation
        explanation.append(f"# {top_path_name}")
        explanation.append(f"## Career Recommendation: {top_path_name}")

        # Summary section
        explanation.append("## Summary")
        explanation.append(f"Based on your profile, the recommended career path is to **{top_path_name}**. Confidence level: **{confidence_level}** ({confidence_score:.1f}/100)")

        if alt_path_name:
            explanation.append(f"\nAlternative option: {alt_path_name}")

        # Expert inputs
        explanation.append("## Expert Inputs")
        for expert, details in results.get("expert_recommendation", {}).get("expert_recommendations", {}).items():
            if details:
                expert_name = expert.replace("_", " ").title()
                explanation.append(f"### {expert_name}")
                # Get the recommendation name from the ID
                rec_id = details.get('recommendation')
                rec_name = next((o["name"] for o in results.get("expert_recommendation", {}).get("career_options", []) if o["id"] == rec_id), rec_id)
                explanation.append(f"Recommendation: {rec_name}")
                explanation.append(f"Rationale: {details.get('explanation', 'Not provided')}")

                if details.get("additional_notes"):
                    explanation.append(f"Additional notes: {details.get('additional_notes')}")

        # Pros and cons
        explanation.append(f"## Pros and Cons of {top_path_name}")

        # Get unique pros and cons to avoid repetition
        option_details = results.get("expert_recommendation", {}).get("option_details", {})
        if top_path_id in option_details:
            pros = option_details[top_path_id].get("pros", [])
            cons = option_details[top_path_id].get("cons", [])
        else:
            pros = []
            cons = []

        # Remove duplicates by description
        unique_pros = []
        unique_pros_descriptions = set()
        for pro in pros:
            desc = pro.get("description", "")
            if desc and desc not in unique_pros_descriptions:
                unique_pros.append(pro)
                unique_pros_descriptions.add(desc)

        unique_cons = []
        unique_cons_descriptions = set()
        for con in cons:
            desc = con.get("description", "")
            if desc and desc not in unique_cons_descriptions:
                unique_cons.append(con)
                unique_cons_descriptions.add(desc)

        explanation.append("### Pros")
        if unique_pros:
            for pro in unique_pros:
                explanation.append(f"- {pro.get('description', 'Not specified')}")
        else:
            explanation.append("- No specific pros identified.")

        explanation.append("### Cons")
        if unique_cons:
            for con in unique_cons:
                explanation.append(f"- {con.get('description', 'Not specified')}")
        else:
            explanation.append("- No specific cons identified.")

        # Simulation results
        explanation.append("## Simulation Results")
        sim_results = results.get("simulation_results", {}).get(top_path_id, {}).get("average_case", {})

        if sim_results:
            explanation.append("The simulation projects the following outcomes over a 5 year period:")
            explanation.append("")

            # Calculate salary growth percentage
            initial_salary = sim_results.get("salaries", [0, 0])[0]
            final_salary = sim_results.get("salaries", [0, 0])[-1]
            salary_growth = ((final_salary - initial_salary) / initial_salary) * 100 if initial_salary > 0 else 0

            explanation.append(f"- Projected salary growth: {salary_growth:.1f}%")

            # Calculate average satisfaction
            satisfactions = sim_results.get("satisfaction", [])
            avg_satisfaction = sum(satisfactions) / len(satisfactions) if satisfactions else 0
            explanation.append(f"- Average job satisfaction: {avg_satisfaction:.1f}/10")

            # Calculate average stability
            stabilities = sim_results.get("stability", [])
            avg_stability = sum(stabilities) / len(stabilities) if stabilities else 0
            explanation.append(f"- Average job stability: {avg_stability:.2f} (1.00 = maximum stability)")

        # Validation
        explanation.append("## Validation")
        validation = results.get("validation", {})

        if validation:
            alignment = validation.get("overall_alignment", 0)
            alignment_text = "strongly" if alignment > 0.8 else "moderately" if alignment > 0.5 else "weakly"

            explanation.append(f"The expert recommendation and simulation results are {alignment_text} aligned.")

            # Alternative paths
            alternatives = validation.get("alternatives", [])
            if alternatives:
                explanation.append("\nPotential alternative paths worth considering based on simulation:")
                explanation.append("")
                for alt in alternatives:
                    explanation.append(f"- {alt.get('name', '')}: {alt.get('reason', '')}")

        # Next steps
        explanation.append("## Next Steps")
        explanation.append("Based on this recommendation, consider:")
        explanation.append("")

        if top_path_id == "stay_bpo":
            explanation.append("- Identifying specific skills to develop within your current role")
            explanation.append("- Setting clear performance goals to stand out in your current position")
            explanation.append("- Networking with colleagues in other departments to explore internal opportunities")
        elif top_path_id == "advance_bpo":
            explanation.append("- Discussing advancement opportunities with your supervisor")
            explanation.append("- Identifying leadership training or specialized skills needed for promotion")
            explanation.append("- Seeking a mentor in a senior role to guide your advancement")
        elif top_path_id == "switch_tech":
            explanation.append("- Researching in-demand tech skills and certifications")
            explanation.append("- Exploring entry-level tech roles that align with your BPO experience")
            explanation.append("- Building a portfolio of tech projects to demonstrate your capabilities")
        elif top_path_id == "switch_business":
            explanation.append("- Identifying transferable skills from BPO to business roles")
            explanation.append("- Researching business certifications or courses to fill skill gaps")
            explanation.append("- Networking with professionals in your target business field")
        elif top_path_id == "further_education":
            explanation.append("- Researching degree programs or certifications that align with your career goals")
            explanation.append("- Exploring part-time or online education options to maintain income")
            explanation.append("- Investigating financial aid or employer tuition assistance programs")

        return "\n".join(explanation)


# --- EXAMPLE USAGE ---

def create_sample_user_profile():
    """Create a sample user profile for demonstration."""
    return {
        "age": 22,
        "bpo_experience_years": 2,
        "current_salary": 28000,  # PHP
        "current_role": "Customer Service Representative",
        "bpo_satisfaction": 8,  # Scale 1-10
        "has_degree": True,
        "highest_education": "Some college",
        "performance_level": "good",  # poor, average, good, excellent
        "interests": ["technology", "leadership"],
        "financial_pressure": "medium",  # low, medium, high
        "personality_traits": {
            "conscientiousness": "low",  # low, medium, high
            "extroversion": "high",
            "openness": "high"
        },
        "identified_alternative_field": False,
        "alternative_field": "tech",
        "researched_requirements": True
    }


def main():
    """Run the system with a sample user profile."""
    # Create the system
    system = CareerTransitionSystem()

    # Create a sample user profile
    user_profile = create_sample_user_profile()

    # Get comprehensive results
    results = system.validate_recommendation(user_profile)

    # Generate explanation
    explanation = system.explain_recommendation(results)
    print(explanation)

    # Optionally save visualization
    if results.get("visualization"):
        results["visualization"].savefig("career_path_comparison.png")
        print("Visualization saved as career_path_comparison.png")


if __name__ == "__main__":
    main()
