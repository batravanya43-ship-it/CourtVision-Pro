"""
Jurisdiction Manager for CourtVision Pro
Handles High Court-specific data processing and local law emphasis
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict

from django.db.models import Q, Count, Avg
from django.core.cache import cache
from django.conf import settings

from .models import Case, HighCourt, Customization
from .search_engine import search_engine

logger = logging.getLogger(__name__)


class JurisdictionManagerError(Exception):
    """Custom exception for jurisdiction manager errors"""
    pass


class HighCourtRuleEngine:
    """Manages High Court-specific rules and procedures"""

    def __init__(self):
        self.court_rules = {}
        self.procedural_codes = {}
        self.load_jurisdiction_rules()

    def load_jurisdiction_rules(self):
        """Load jurisdiction-specific rules and procedures"""
        try:
            # Define rules for major High Courts
            self.court_rules = {
                'Delhi High Court': {
                    'commercial_court_division': True,
                    'fast_track_procedures': True,
                    'mediation_mandatory': True,
                    'document_filing_format': 'digital',
                    'case_time_limit_days': 180,
                    'appeal_period_days': 30,
                    'specialized_judges': True,
                    'local_acts': ['Delhi Commercial Courts Act', 'Delhi Rent Control Act'],
                    'procedural_preferences': {
                        'emphasis': 'speedy_resolution',
                        'documentation': 'extensive',
                        'evidence': 'digital_preferred'
                    }
                },
                'Bombay High Court': {
                    'commercial_court_division': True,
                    'fast_track_procedures': True,
                    'mediation_mandatory': False,
                    'document_filing_format': 'hybrid',
                    'case_time_limit_days': 240,
                    'appeal_period_days': 45,
                    'specialized_judges': True,
                    'local_acts': ['Maharashtra Rent Control Act', 'Bombay Stamp Act'],
                    'procedural_preferences': {
                        'emphasis': 'thorough_analysis',
                        'documentation': 'detailed',
                        'evidence': 'traditional_preferred'
                    }
                },
                'Calcutta High Court': {
                    'commercial_court_division': True,
                    'fast_track_procedures': False,
                    'mediation_mandatory': True,
                    'document_filing_format': 'traditional',
                    'case_time_limit_days': 300,
                    'appeal_period_days': 60,
                    'specialized_judges': False,
                    'local_acts': ['West Bengal Commercial Courts Act', 'West Bengal Land Reforms Act'],
                    'procedural_preferences': {
                        'emphasis': 'legal_precision',
                        'documentation': 'comprehensive',
                        'evidence': 'traditional_required'
                    }
                },
                'Madras High Court': {
                    'commercial_court_division': True,
                    'fast_track_procedures': True,
                    'mediation_mandatory': True,
                    'document_filing_format': 'digital',
                    'case_time_limit_days': 210,
                    'appeal_period_days': 30,
                    'specialized_judges': True,
                    'local_acts': ['Tamil Nadu Commercial Courts Act', 'Tamil Nadu Buildings Act'],
                    'procedural_preferences': {
                        'emphasis': 'technology_integration',
                        'documentation': 'digital_preferred',
                        'evidence': 'digital_accepted'
                    }
                }
            }

            # Load procedural codes
            self.procedural_codes = {
                'commercial_disputes': {
                    'limit': 'value_based',
                    'procedure': 'summary_judgment',
                    'evidence': 'document_heavy'
                },
                'civil_suits': {
                    'limit': 'jurisdiction_based',
                    'procedure': 'standard',
                    'evidence': 'balanced'
                },
                'corporate_matters': {
                    'limit': 'specialized',
                    'procedure': 'expedited',
                    'evidence': 'expert_heavy'
                }
            }

            logger.info("Jurisdiction rules loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load jurisdiction rules: {str(e)}")

    def get_court_rules(self, court_name: str) -> Dict[str, Any]:
        """Get rules for a specific court"""
        return self.court_rules.get(court_name, self._get_default_rules())

    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default rules for unknown courts"""
        return {
            'commercial_court_division': False,
            'fast_track_procedures': False,
            'mediation_mandatory': False,
            'document_filing_format': 'traditional',
            'case_time_limit_days': 365,
            'appeal_period_days': 90,
            'specialized_judges': False,
            'local_acts': [],
            'procedural_preferences': {
                'emphasis': 'standard_procedure',
                'documentation': 'standard',
                'evidence': 'balanced'
            }
        }

    def apply_jurisdiction_filtering(self, search_results: List[Dict], court_name: str,
                                   user_customization: Optional[Customization] = None) -> List[Dict]:
        """Apply jurisdiction-specific filtering to search results"""
        try:
            court_rules = self.get_court_rules(court_name)
            filtered_results = []

            for result in search_results:
                relevance_score = result.get('relevance_score', 0)

                # Boost local cases
                if result.get('court') == court_name:
                    relevance_score *= 1.5

                # Boost cases with local acts
                local_acts = court_rules.get('local_acts', [])
                case_statutes = result.get('statutes_cited', [])
                if any(act in str(case_statutes) for act in local_acts):
                    relevance_score *= 1.3

                # Apply procedural preferences
                procedural_prefs = court_rules.get('procedural_preferences', {})
                if self._matches_procedural_preferences(result, procedural_prefs):
                    relevance_score *= 1.2

                # Update result
                result['jurisdiction_boosted_score'] = relevance_score
                result['jurisdiction_factors'] = {
                    'local_court': result.get('court') == court_name,
                    'local_acts_cited': any(act in str(case_statutes) for act in local_acts),
                    'procedural_match': self._matches_procedural_preferences(result, procedural_prefs)
                }

                filtered_results.append(result)

            # Sort by boosted scores
            filtered_results.sort(key=lambda x: x['jurisdiction_boosted_score'], reverse=True)

            return filtered_results

        except Exception as e:
            logger.error(f"Jurisdiction filtering failed: {str(e)}")
            return search_results

    def _matches_procedural_preferences(self, result: Dict, preferences: Dict) -> bool:
        """Check if case matches procedural preferences"""
        # Simplified matching logic
        emphasis = preferences.get('emphasis', 'standard')

        if emphasis == 'speedy_resolution':
            # Prefer cases with shorter duration
            return result.get('view_count', 0) > 10  # Proxy for well-established cases

        elif emphasis == 'thorough_analysis':
            # Prefer cases with extensive citations
            statutes = result.get('statutes_cited', [])
            precedents = result.get('precedents_cited', [])
            return len(statutes) > 5 or len(precedents) > 10

        elif emphasis == 'legal_precision':
            # Prefer cases with formal legal language
            return any(term in result.get('snippet', '').lower()
                      for term in ['whereas', 'therefore', 'pursuant', 'notwithstanding'])

        return True

    def get_jurisdiction_specific_guidance(self, court_name: str, case_type: str) -> Dict[str, Any]:
        """Get jurisdiction-specific procedural guidance"""
        try:
            court_rules = self.get_court_rules(court_name)
            procedural_code = self.procedural_codes.get(case_type, {})

            guidance = {
                'court_name': court_name,
                'case_type': case_type,
                'procedural_requirements': {
                    'fast_track_available': court_rules.get('fast_track_procedures', False),
                    'mediation_required': court_rules.get('mediation_mandatory', False),
                    'specialized_division': court_rules.get('commercial_court_division', False),
                    'document_format': court_rules.get('document_filing_format', 'traditional')
                },
                'timeline_information': {
                    'expected_duration_days': court_rules.get('case_time_limit_days', 365),
                    'appeal_period_days': court_rules.get('appeal_period_days', 90),
                    'fast_track_reduction': 0.5 if court_rules.get('fast_track_procedures') else 1.0
                },
                'local_legislation': court_rules.get('local_acts', []),
                'procedural_tips': self._generate_procedural_tips(court_rules, case_type)
            }

            return guidance

        except Exception as e:
            logger.error(f"Failed to generate jurisdiction guidance: {str(e)}")
            return {}

    def _generate_procedural_tips(self, court_rules: Dict, case_type: str) -> List[str]:
        """Generate procedural tips for the jurisdiction"""
        tips = []

        if court_rules.get('fast_track_procedures'):
            tips.append("Consider applying for fast-track proceedings to expedite resolution")

        if court_rules.get('mediation_mandatory'):
            tips.append("Mediation is mandatory - prepare settlement proposals")

        if court_rules.get('commercial_court_division'):
            tips.append("Case will be heard in specialized commercial division")

        doc_format = court_rules.get('document_filing_format', 'traditional')
        if doc_format == 'digital':
            tips.append("Ensure all documents are in digital format as per court requirements")
        elif doc_format == 'hybrid':
            tips.append("Both digital and physical documents accepted - prepare both formats")

        local_acts = court_rules.get('local_acts', [])
        if local_acts:
            tips.append(f"Be aware of applicable local legislation: {', '.join(local_acts[:3])}")

        return tips


class LocalEmphasisEngine:
    """Engine for emphasizing local laws and procedures"""

    def __init__(self):
        self.rule_engine = HighCourtRuleEngine()
        self.local_precedent_weights = {}
        self.load_local_precedent_data()

    def load_local_precedent_data(self):
        """Load local precedent weighting data"""
        try:
            # Define local precedent importance by court
            self.local_precedent_weights = {
                'Delhi High Court': {
                    'delhi_high_court_cases': 2.0,
                    'supreme_court_cases': 1.8,
                    'other_high_courts': 1.2,
                    'international_cases': 0.8
                },
                'Bombay High Court': {
                    'bombay_high_court_cases': 2.0,
                    'supreme_court_cases': 1.7,
                    'other_high_courts': 1.3,
                    'international_cases': 0.9
                },
                'Calcutta High Court': {
                    'calcutta_high_court_cases': 2.0,
                    'supreme_court_cases': 1.9,
                    'other_high_courts': 1.1,
                    'international_cases': 0.7
                },
                'Madras High Court': {
                    'madras_high_court_cases': 2.0,
                    'supreme_court_cases': 1.8,
                    'other_high_courts': 1.2,
                    'international_cases': 0.8
                }
            }

            logger.info("Local precedent data loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load local precedent data: {str(e)}")

    def apply_local_emphasis(self, search_results: List[Dict], court_name: str,
                           user_preferences: Optional[Dict] = None) -> List[Dict]:
        """Apply local emphasis to search results"""
        try:
            # Get jurisdiction rules
            court_rules = self.rule_engine.get_court_rules(court_name)
            precedent_weights = self.local_precedent_weights.get(court_name, {})

            emphasized_results = []

            for result in search_results:
                original_score = result.get('relevance_score', 0)
                emphasized_score = original_score

                # Apply court-specific weighting
                result_court = result.get('court', '')
                if 'delhi' in result_court.lower() and court_name == 'Delhi High Court':
                    weight = precedent_weights.get('delhi_high_court_cases', 1.0)
                    emphasized_score *= weight
                elif 'bombay' in result_court.lower() and court_name == 'Bombay High Court':
                    weight = precedent_weights.get('bombay_high_court_cases', 1.0)
                    emphasized_score *= weight
                elif 'calcutta' in result_court.lower() and court_name == 'Calcutta High Court':
                    weight = precedent_weights.get('calcutta_high_court_cases', 1.0)
                    emphasized_score *= weight
                elif 'madras' in result_court.lower() and court_name == 'Madras High Court':
                    weight = precedent_weights.get('madras_high_court_cases', 1.0)
                    emphasized_score *= weight
                elif 'supreme court' in result_court.lower():
                    weight = precedent_weights.get('supreme_court_cases', 1.0)
                    emphasized_score *= weight
                elif 'high court' in result_court.lower():
                    weight = precedent_weights.get('other_high_courts', 1.0)
                    emphasized_score *= weight

                # Apply user preference emphasis
                if user_preferences:
                    emphasized_score = self._apply_user_preference_emphasis(
                        result, emphasized_score, user_preferences
                    )

                # Update result with emphasis data
                result['emphasized_score'] = emphasized_score
                result['emphasis_applied'] = True
                result['emphasis_factors'] = {
                    'local_court_boost': result_court.lower().find(court_name.split()[0].lower()) != -1,
                    'precedent_weight': emphasized_score / original_score if original_score > 0 else 1.0
                }

                emphasized_results.append(result)

            # Re-sort by emphasized scores
            emphasized_results.sort(key=lambda x: x['emphasized_score'], reverse=True)

            return emphasized_results

        except Exception as e:
            logger.error(f"Local emphasis application failed: {str(e)}")
            return search_results

    def _apply_user_preference_emphasis(self, result: Dict, score: float,
                                      preferences: Dict) -> float:
        """Apply user-specific preference emphasis"""
        emphasized_score = score

        # Time period emphasis
        time_preference = preferences.get('time_period_focus', 'recent')
        if time_preference == 'recent':
            # Boost recent cases
            judgment_date = result.get('judgment_date', '')
            if judgment_date:
                case_date = datetime.fromisoformat(judgment_date.replace('Z', '+00:00'))
                if datetime.now(case_date.tzinfo) - case_date < timedelta(days=365*5):
                    emphasized_score *= 1.2

        # Legal emphasis
        legal_emphasis = preferences.get('legal_emphasis', 'balanced')
        if legal_emphasis == 'precedent':
            # Boost cases with many precedents
            precedents_count = len(result.get('precedents_cited', []))
            if precedents_count > 10:
                emphasized_score *= 1.1
        elif legal_emphasis == 'statute':
            # Boost cases with statutory citations
            statutes_count = len(result.get('statutes_cited', []))
            if statutes_count > 5:
                emphasized_score *= 1.1

        return emphasized_score

    def get_local_context_summary(self, court_name: str, case_type: str) -> Dict[str, Any]:
        """Get summary of local legal context"""
        try:
            # Get recent local cases
            court = HighCourt.objects.filter(name__icontains=court_name.split()[0]).first()
            if not court:
                return {'error': 'Court not found'}

            recent_cases = Case.objects.filter(
                court=court,
                case_type=case_type,
                judgment_date__gte=datetime.now() - timedelta(days=365*2)
            ).select_related('court').prefetch_related('tags')[:20]

            # Analyze local patterns
            local_patterns = self._analyze_local_patterns(recent_cases)

            # Get jurisdiction guidance
            guidance = self.rule_engine.get_jurisdiction_specific_guidance(court.name, case_type)

            return {
                'court_name': court.name,
                'case_type': case_type,
                'local_patterns': local_patterns,
                'jurisdiction_guidance': guidance,
                'recent_local_cases': [
                    {
                        'title': case.title,
                        'citation': case.citation,
                        'judgment_date': case.judgment_date.isoformat(),
                        'key_points': case.ai_summary.get('key_points', [])[:3] if case.ai_summary else []
                    }
                    for case in recent_cases[:10]
                ],
                'context_summary': self._generate_context_summary(local_patterns, guidance)
            }

        except Exception as e:
            logger.error(f"Failed to generate local context summary: {str(e)}")
            return {'error': str(e)}

    def _analyze_local_patterns(self, cases) -> Dict[str, Any]:
        """Analyze patterns in local cases"""
        if not cases:
            return {}

        # Common outcomes
        outcomes = []
        for case in cases:
            if case.ai_summary and isinstance(case.ai_summary, dict):
                decision = case.ai_summary.get('decision', '').lower()
                if 'allowed' in decision:
                    outcomes.append('petitioner_favorable')
                elif 'dismissed' in decision:
                    outcomes.append('respondent_favorable')
                else:
                    outcomes.append('other')

        # Common tags
        tag_counts = defaultdict(int)
        for case in cases:
            for tag in case.tags.all():
                tag_counts[tag.name] += 1

        # Average duration
        durations = []
        for case in cases:
            duration = (case.decision_date - case.judgment_date).days
            durations.append(duration)

        return {
            'common_outcomes': dict(Counter(outcomes)),
            'common_tags': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'average_duration_days': sum(durations) / len(durations) if durations else 0,
            'total_cases_analyzed': len(cases)
        }

    def _generate_context_summary(self, patterns: Dict, guidance: Dict) -> str:
        """Generate a narrative summary of local context"""
        summary_parts = []

        # Add outcome patterns
        if patterns.get('common_outcomes'):
            most_common = max(patterns['common_outcomes'].items(), key=lambda x: x[1])
            summary_parts.append(f"Most common outcome: {most_common[0]} ({most_common[1]}% of cases)")

        # Add duration information
        if patterns.get('average_duration_days'):
            avg_days = int(patterns['average_duration_days'])
            summary_parts.append(f"Average case duration: {avg_days} days")

        # Add procedural information
        procedural_reqs = guidance.get('procedural_requirements', {})
        if procedural_reqs.get('fast_track_available'):
            summary_parts.append("Fast-track procedures are available")
        if procedural_reqs.get('mediation_required'):
            summary_parts.append("Mediation is mandatory")

        return " | ".join(summary_parts)


# Global instances
jurisdiction_manager = HighCourtRuleEngine()
local_emphasis_engine = LocalEmphasisEngine()


def apply_jurisdiction_processing(search_results: List[Dict], court_name: str,
                                 user_customization: Optional[Customization] = None) -> List[Dict]:
    """Apply complete jurisdiction processing to search results"""
    try:
        # Apply jurisdiction filtering
        filtered_results = jurisdiction_manager.apply_jurisdiction_filtering(
            search_results, court_name, user_customization
        )

        # Apply local emphasis
        emphasized_results = local_emphasis_engine.apply_local_emphasis(
            filtered_results, court_name,
            user_customization.jurisdiction_emphasis if user_customization else None
        )

        return emphasized_results

    except Exception as e:
        logger.error(f"Jurisdiction processing failed: {str(e)}")
        return search_results


def get_jurisdiction_insights(court_name: str, case_type: str) -> Dict[str, Any]:
    """Get comprehensive jurisdiction insights"""
    try:
        # Get local context
        local_context = local_emphasis_engine.get_local_context_summary(court_name, case_type)

        # Get jurisdiction guidance
        guidance = jurisdiction_manager.get_jurisdiction_specific_guidance(court_name, case_type)

        return {
            'local_context': local_context,
            'procedural_guidance': guidance,
            'recommendations': generate_jurisdiction_recommendations(guidance, local_context),
            'insight_date': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get jurisdiction insights: {str(e)}")
        return {'error': str(e)}


def generate_jurisdiction_recommendations(guidance: Dict, local_context: Dict) -> List[str]:
    """Generate recommendations based on jurisdiction analysis"""
    recommendations = []

    try:
        procedural_reqs = guidance.get('procedural_requirements', {})
        patterns = local_context.get('local_patterns', {})

        # Fast-track recommendations
        if procedural_reqs.get('fast_track_available'):
            recommendations.append("Consider filing for fast-track proceedings to reduce resolution time")

        # Mediation recommendations
        if procedural_reqs.get('mediation_required'):
            recommendations.append("Prepare for mandatory mediation process")

        # Document format recommendations
        doc_format = procedural_reqs.get('document_format', 'traditional')
        if doc_format == 'digital':
            recommendations.append("Ensure all documentation is in digital format as per court requirements")

        # Duration-based recommendations
        avg_duration = patterns.get('average_duration_days', 0)
        if avg_duration > 200:
            recommendations.append("Cases in this jurisdiction typically take longer than average - plan accordingly")
        elif avg_duration < 100:
            recommendations.append("This jurisdiction has relatively quick resolution times")

        # Local legislation recommendations
        local_acts = guidance.get('local_legislation', [])
        if local_acts:
            recommendations.append(f"Pay special attention to: {', '.join(local_acts[:2])}")

    except Exception as e:
        logger.error(f"Failed to generate recommendations: {str(e)}")

    return recommendations