"""
Ethical AI Framework for CourtVision Pro
Ensures ethical AI practices with transparency, bias detection, and human oversight
"""

import json
import logging
import hashlib
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics

from django.db import models
from django.core.cache import cache
from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone

try:
    import shap
    import lime
    import lime.lime_text
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .models import Case, UserProfile

logger = logging.getLogger(__name__)


class EthicsFrameworkError(Exception):
    """Custom exception for ethics framework errors"""
    pass


class BiasDetectionEngine:
    """Detects and mitigates bias in AI predictions and recommendations"""

    def __init__(self):
        self.protected_attributes = ['gender', 'religion', 'caste', 'region', 'language']
        self.bias_thresholds = {
            'demographic_parity_difference': 0.1,
            'equal_opportunity_difference': 0.1,
            'disparate_impact_ratio': 0.8
        }
        self.bias_metrics_cache = {}
        self.cache_timeout = 3600  # 1 hour

    def detect_prediction_bias(self, predictions: List[Dict], demographic_data: List[Dict]) -> Dict[str, Any]:
        """Detect bias in AI predictions across demographic groups"""
        try:
            if not predictions or not demographic_data:
                return {'error': 'Insufficient data for bias detection'}

            # Group predictions by demographic attributes
            demographic_groups = self._group_by_demographics(predictions, demographic_data)

            # Calculate bias metrics
            bias_metrics = {}

            for attribute in self.protected_attributes:
                if attribute in demographic_groups:
                    attribute_metrics = self._calculate_attribute_bias(
                        demographic_groups[attribute], attribute
                    )
                    bias_metrics[attribute] = attribute_metrics

            # Overall bias assessment
            overall_bias = self._assess_overall_bias(bias_metrics)

            # Generate recommendations
            recommendations = self._generate_bias_mitigation_recommendations(bias_metrics)

            return {
                'bias_detected': overall_bias['bias_detected'],
                'overall_bias_score': overall_bias['bias_score'],
                'bias_metrics': bias_metrics,
                'recommendations': recommendations,
                'analysis_date': datetime.now().isoformat(),
                'sample_size': len(predictions),
                'demographic_coverage': self._calculate_demographic_coverage(demographic_data)
            }

        except Exception as e:
            logger.error(f"Bias detection failed: {str(e)}")
            return {'error': str(e)}

    def _group_by_demographics(self, predictions: List[Dict], demographic_data: List[Dict]) -> Dict[str, Dict]:
        """Group predictions by demographic attributes"""
        groups = defaultdict(lambda: defaultdict(list))

        for i, (prediction, demographic) in enumerate(zip(predictions, demographic_data)):
            if i >= len(predictions) or i >= len(demographic_data):
                continue

            for attribute in self.protected_attributes:
                if attribute in demographic:
                    value = demographic[attribute]
                    groups[attribute][value].append(prediction)

        return dict(groups)

    def _calculate_attribute_bias(self, group_data: Dict, attribute: str) -> Dict[str, Any]:
        """Calculate bias metrics for a specific demographic attribute"""
        try:
            if len(group_data) < 2:
                return {'error': 'Insufficient groups for comparison'}

            # Calculate outcome rates for each group
            group_outcomes = {}
            for group_value, predictions in group_data.items():
                positive_outcomes = sum(
                    1 for p in predictions
                    if self._extract_positive_outcome(p)
                )
                outcome_rate = positive_outcomes / len(predictions) if predictions else 0
                group_outcomes[group_value] = {
                    'outcome_rate': outcome_rate,
                    'sample_size': len(predictions),
                    'positive_outcomes': positive_outcomes
                }

            # Calculate bias metrics
            bias_metrics = {
                'group_outcomes': group_outcomes,
                'demographic_parity_difference': self._calculate_demographic_parity(group_outcomes),
                'equal_opportunity_difference': self._calculate_equal_opportunity(group_outcomes),
                'disparate_impact_ratio': self._calculate_disparate_impact(group_outcomes)
            }

            # Determine if bias is detected
            bias_detected = any(
                abs(metric) > self.bias_thresholds.get(threshold, 0.1)
                for threshold, metric in [
                    ('demographic_parity_difference', bias_metrics['demographic_parity_difference']),
                    ('equal_opportunity_difference', bias_metrics['equal_opportunity_difference'])
                ]
                if metric is not None
            )

            bias_metrics['bias_detected'] = bias_detected

            return bias_metrics

        except Exception as e:
            logger.error(f"Attribute bias calculation failed: {str(e)}")
            return {'error': str(e)}

    def _extract_positive_outcome(self, prediction: Dict) -> bool:
        """Extract whether prediction indicates a positive outcome"""
        try:
            outcome = prediction.get('predicted_outcome', '').lower()
            confidence = prediction.get('confidence', 0)

            # Consider high-confidence favorable outcomes as positive
            positive_outcomes = ['allowed', 'granted', 'favorable', 'successful']
            return any(pos in outcome for pos in positive_outcomes) and confidence > 0.7

        except Exception:
            return False

    def _calculate_demographic_parity(self, group_outcomes: Dict) -> Optional[float]:
        """Calculate demographic parity difference"""
        try:
            rates = [data['outcome_rate'] for data in group_outcomes.values()]
            if len(rates) < 2:
                return None

            return max(rates) - min(rates)

        except Exception:
            return None

    def _calculate_equal_opportunity(self, group_outcomes: Dict) -> Optional[float]:
        """Calculate equal opportunity difference (simplified)"""
        try:
            rates = [data['outcome_rate'] for data in group_outcomes.values()]
            if len(rates) < 2:
                return None

            return max(rates) - min(rates)

        except Exception:
            return None

    def _calculate_disparate_impact(self, group_outcomes: Dict) -> Optional[float]:
        """Calculate disparate impact ratio"""
        try:
            rates = [data['outcome_rate'] for data in group_outcomes.values() if data['outcome_rate'] > 0]
            if len(rates) < 2:
                return None

            return min(rates) / max(rates)

        except Exception:
            return None

    def _assess_overall_bias(self, bias_metrics: Dict) -> Dict[str, Any]:
        """Assess overall bias across all attributes"""
        try:
            bias_detected_count = 0
            total_attributes = 0
            bias_scores = []

            for attribute, metrics in bias_metrics.items():
                if 'error' not in metrics:
                    total_attributes += 1
                    if metrics.get('bias_detected', False):
                        bias_detected_count += 1

                    # Calculate bias score for this attribute
                    parity_diff = metrics.get('demographic_parity_difference', 0)
                    if parity_diff is not None:
                        bias_scores.append(abs(parity_diff))

            overall_bias_score = statistics.mean(bias_scores) if bias_scores else 0

            return {
                'bias_detected': bias_detected_count > 0,
                'bias_score': overall_bias_score,
                'biased_attributes_count': bias_detected_count,
                'total_attributes_analyzed': total_attributes,
                'bias_percentage': (bias_detected_count / total_attributes * 100) if total_attributes > 0 else 0
            }

        except Exception as e:
            logger.error(f"Overall bias assessment failed: {str(e)}")
            return {'bias_detected': False, 'bias_score': 0, 'error': str(e)}

    def _generate_bias_mitigation_recommendations(self, bias_metrics: Dict) -> List[str]:
        """Generate recommendations to mitigate detected bias"""
        recommendations = []

        try:
            for attribute, metrics in bias_metrics.items():
                if 'error' in metrics or not metrics.get('bias_detected', False):
                    continue

                recommendations.append(f"Review and mitigate bias in {attribute} attribute")

                # Specific recommendations based on bias type
                parity_diff = metrics.get('demographic_parity_difference', 0)
                if parity_diff and abs(parity_diff) > 0.1:
                    recommendations.append(f"Consider re-balancing training data for {attribute}")

                disparate_impact = metrics.get('disparate_impact_ratio', 1.0)
                if disparate_impact and disparate_impact < 0.8:
                    recommendations.append(f"Implement fairness constraints for {attribute} predictions")

            if not recommendations:
                recommendations.append("No significant bias detected - continue monitoring")

        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")

        return recommendations

    def _calculate_demographic_coverage(self, demographic_data: List[Dict]) -> Dict[str, Any]:
        """Calculate coverage of demographic attributes in data"""
        try:
            coverage = {}
            total_records = len(demographic_data)

            for attribute in self.protected_attributes:
                present_count = sum(1 for d in demographic_data if attribute in d and d[attribute])
                coverage[attribute] = {
                    'present_count': present_count,
                    'total_count': total_records,
                    'coverage_percentage': (present_count / total_records * 100) if total_records > 0 else 0
                }

            return coverage

        except Exception as e:
            logger.error(f"Demographic coverage calculation failed: {str(e)}")
            return {}


class ExplainabilityEngine:
    """Provides explanations for AI predictions and recommendations"""

    def __init__(self):
        self.explanation_cache = {}
        self.cache_timeout = 1800  # 30 minutes
        self.lime_explainer = None
        self.initialize_explainers()

    def initialize_explainers(self):
        """Initialize explanation models"""
        try:
            if SHAP_AVAILABLE:
                # Initialize SHAP explainer
                pass  # Would be initialized with actual model

            # Initialize LIME explainer for text
            if SHAP_AVAILABLE:
                self.lime_explainer = lime.lime_text.LimeTextExplainer(
                    class_names=['unfavorable', 'favorable']
                )

            logger.info("Explanation engines initialized")

        except Exception as e:
            logger.error(f"Explainer initialization failed: {str(e)}")

    def generate_prediction_explanation(self, prediction: Dict, case_context: Dict,
                                      model_info: Dict) -> Dict[str, Any]:
        """Generate explanation for a specific prediction"""
        try:
            explanation_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            # Generate different types of explanations
            feature_explanation = self._generate_feature_explanation(prediction, case_context)
            confidence_explanation = self._generate_confidence_explanation(prediction)
            precedent_explanation = self._generate_precedent_explanation(case_context)
            limitation_explanation = self._generate_limitation_explanation(prediction, model_info)

            explanation = {
                'explanation_id': explanation_id,
                'prediction_id': prediction.get('case_id', 'unknown'),
                'explanation_type': 'comprehensive',
                'timestamp': timestamp,
                'feature_importance': feature_explanation,
                'confidence_breakdown': confidence_explanation,
                'precedent_analysis': precedent_explanation,
                'limitations_and_caveats': limitation_explanation,
                'human_review_required': self._assess_human_review_need(prediction),
                'explanation_quality': self._assess_explanation_quality(prediction, case_context)
            }

            # Cache explanation
            cache_key = f"explanation_{prediction.get('case_id', 'unknown')}"
            cache.set(cache_key, explanation, timeout=self.cache_timeout)

            return explanation

        except Exception as e:
            logger.error(f"Prediction explanation generation failed: {str(e)}")
            return {
                'error': str(e),
                'prediction_id': prediction.get('case_id', 'unknown'),
                'explanation_failed': True
            }

    def _generate_feature_explanation(self, prediction: Dict, case_context: Dict) -> Dict[str, Any]:
        """Generate feature importance explanation"""
        try:
            # Simplified feature importance based on available data
            features = {}

            # Case type importance
            case_type = case_context.get('case_type', '')
            if case_type:
                features['case_type'] = {
                    'value': case_type,
                    'importance': 0.3,
                    'explanation': f"Cases of type '{case_type}' have specific procedural requirements"
                }

            # Court importance
            court = case_context.get('court', '')
            if court:
                features['court'] = {
                    'value': court,
                    'importance': 0.25,
                    'explanation': f"{court} has specific procedural preferences and precedents"
                }

            # Tags importance
            tags = case_context.get('tags', [])
            if tags:
                features['legal_topics'] = {
                    'value': ', '.join(tags[:3]),
                    'importance': 0.2,
                    'explanation': "Legal topics indicate relevant precedents and applicable laws"
                }

            # Time factors
            judgment_date = case_context.get('judgment_date')
            if judgment_date:
                features['temporal_factors'] = {
                    'value': 'Recent case',
                    'importance': 0.15,
                    'explanation': 'Recent cases may reflect current legal trends'
                }

            # Complexity factors
            text_length = len(case_context.get('case_text', ''))
            if text_length > 0:
                complexity = 'high' if text_length > 10000 else 'medium' if text_length > 5000 else 'low'
                features['case_complexity'] = {
                    'value': complexity,
                    'importance': 0.1,
                    'explanation': f'Case complexity ({complexity}) affects prediction confidence'
                }

            return {
                'method': 'rule_based',
                'features': features,
                'total_importance': sum(f['importance'] for f in features.values())
            }

        except Exception as e:
            logger.error(f"Feature explanation generation failed: {str(e)}")
            return {'error': str(e), 'method': 'failed'}

    def _generate_confidence_explanation(self, prediction: Dict) -> Dict[str, Any]:
        """Generate confidence breakdown explanation"""
        try:
            confidence = prediction.get('confidence', 0)
            outcome = prediction.get('predicted_outcome', 'unknown')

            # Confidence factors
            factors = []

            # Base confidence from model
            factors.append({
                'factor': 'Model Confidence',
                'value': confidence,
                'description': 'Base confidence from machine learning model',
                'weight': 0.6
            })

            # Historical precedent strength
            precedent_strength = prediction.get('precedent_strength', 0.5)
            factors.append({
                'factor': 'Precedent Strength',
                'value': precedent_strength,
                'description': 'Strength and relevance of supporting precedents',
                'weight': 0.25
            })

            # Data quality
            data_quality = prediction.get('data_quality', 0.7)
            factors.append({
                'factor': 'Data Quality',
                'value': data_quality,
                'description': 'Quality and completeness of case data',
                'weight': 0.15
            })

            # Calculate weighted confidence
            weighted_confidence = sum(f['value'] * f['weight'] for f in factors)

            # Confidence level categorization
            if weighted_confidence >= 0.8:
                confidence_level = 'high'
                reliability = 'highly reliable'
            elif weighted_confidence >= 0.6:
                confidence_level = 'medium'
                reliability = 'moderately reliable'
            else:
                confidence_level = 'low'
                reliability = 'use with caution'

            return {
                'overall_confidence': round(weighted_confidence, 3),
                'confidence_level': confidence_level,
                'reliability_assessment': reliability,
                'contributing_factors': factors,
                'predicted_outcome': outcome
            }

        except Exception as e:
            logger.error(f"Confidence explanation generation failed: {str(e)}")
            return {'error': str(e)}

    def _generate_precedent_explanation(self, case_context: Dict) -> Dict[str, Any]:
        """Generate precedent analysis explanation"""
        try:
            precedents = case_context.get('precedents_cited', [])
            statutes = case_context.get('statutes_cited', [])

            precedent_analysis = {
                'total_precedents': len(precedents),
                'total_statutes': len(statutes),
                'precedent_strength': 'strong' if len(precedents) > 10 else 'moderate' if len(precedents) > 5 else 'limited',
                'legal_basis': []
            }

            # Analyze precedent types
            if precedents:
                precedent_analysis['legal_basis'].append(
                    f"Based on {len(precedents)} relevant precedents"
                )

            if statutes:
                precedent_analysis['legal_basis'].append(
                    f"Supported by {len(statutes)} statutory provisions"
                )

            # Jurisdiction relevance
            court = case_context.get('court', '')
            if court:
                precedent_analysis['legal_basis'].append(
                    f"Considers {court} jurisdiction-specific precedents"
                )

            if not precedent_analysis['legal_basis']:
                precedent_analysis['legal_basis'].append('Limited legal precedent available')

            return precedent_analysis

        except Exception as e:
            logger.error(f"Precedent explanation generation failed: {str(e)}")
            return {'error': str(e)}

    def _generate_limitation_explanation(self, prediction: Dict, model_info: Dict) -> Dict[str, Any]:
        """Generate limitations and caveats explanation"""
        try:
            limitations = []

            # Model limitations
            model_version = model_info.get('version', 'unknown')
            limitations.append({
                'type': 'model_limitations',
                'description': f"Predictions based on model version {model_version}",
                'impact': 'Model performance may vary with different case types'
            })

            # Data limitations
            training_data_size = model_info.get('training_data_size', 0)
            if training_data_size < 1000:
                limitations.append({
                    'type': 'data_limitations',
                    'description': f"Limited training data ({training_data_size} cases)",
                    'impact': 'Predictions may be less reliable for rare case types'
                })

            # Confidence limitations
            confidence = prediction.get('confidence', 0)
            if confidence < 0.7:
                limitations.append({
                    'type': 'confidence_limitations',
                    'description': f"Low confidence prediction ({confidence:.2f})",
                    'impact': 'Human review strongly recommended'
                })

            # Temporal limitations
            model_training_date = model_info.get('training_date')
            if model_training_date:
                training_date = datetime.fromisoformat(model_training_date)
                days_old = (datetime.now() - training_date).days
                if days_old > 90:
                    limitations.append({
                        'type': 'temporal_limitations',
                        'description': f"Model trained {days_old} days ago",
                        'impact': 'May not reflect recent legal developments'
                    })

            return {
                'limitations': limitations,
                'overall_reliability': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
                'human_review_recommended': len(limitations) > 2 or confidence < 0.7
            }

        except Exception as e:
            logger.error(f"Limitation explanation generation failed: {str(e)}")
            return {'error': str(e)}

    def _assess_human_review_need(self, prediction: Dict) -> Dict[str, Any]:
        """Assess if human review is needed"""
        try:
            factors = []

            confidence = prediction.get('confidence', 0)
            if confidence < 0.6:
                factors.append('Low confidence score')

            if prediction.get('prediction_failed', False):
                factors.append('Prediction processing failed')

            if prediction.get('novel_case', False):
                factors.append('Novel case type')

            # Check for edge cases
            if prediction.get('edge_case_detected', False):
                factors.append('Edge case detected')

            human_review_needed = len(factors) > 0
            urgency = 'high' if confidence < 0.4 else 'medium' if confidence < 0.7 else 'low'

            return {
                'review_required': human_review_needed,
                'urgency': urgency,
                'reasoning': factors,
                'recommended_reviewer': 'judicial_officer' if human_review_needed else None
            }

        except Exception as e:
            logger.error(f"Human review assessment failed: {str(e)}")
            return {'review_required': True, 'urgency': 'high', 'error': str(e)}

    def _assess_explanation_quality(self, prediction: Dict, case_context: Dict) -> Dict[str, Any]:
        """Assess the quality of generated explanation"""
        try:
            quality_factors = []

            # Data completeness
            if case_context.get('case_text'):
                quality_factors.append({'factor': 'complete_case_data', 'score': 1.0})
            else:
                quality_factors.append({'factor': 'complete_case_data', 'score': 0.3})

            # Precedent availability
            if case_context.get('precedents_cited'):
                quality_factors.append({'factor': 'precedent_available', 'score': 1.0})
            else:
                quality_factors.append({'factor': 'precedent_available', 'score': 0.5})

            # Confidence level
            confidence = prediction.get('confidence', 0)
            if confidence > 0.8:
                quality_factors.append({'factor': 'high_confidence', 'score': 1.0})
            elif confidence > 0.6:
                quality_factors.append({'factor': 'medium_confidence', 'score': 0.7})
            else:
                quality_factors.append({'factor': 'low_confidence', 'score': 0.4})

            # Calculate overall quality score
            overall_score = sum(f['score'] for f in quality_factors) / len(quality_factors)

            # Quality rating
            if overall_score >= 0.8:
                quality_rating = 'high'
                reliability = 'highly reliable'
            elif overall_score >= 0.6:
                quality_rating = 'medium'
                reliability = 'moderately reliable'
            else:
                quality_rating = 'low'
                reliability = 'use with caution'

            return {
                'overall_quality_score': round(overall_score, 3),
                'quality_rating': quality_rating,
                'reliability': reliability,
                'quality_factors': quality_factors
            }

        except Exception as e:
            logger.error(f"Explanation quality assessment failed: {str(e)}")
            return {'quality_rating': 'unknown', 'error': str(e)}


class AuditLogManager:
    """Manages audit logging for all AI decisions and processes"""

    def __init__(self):
        self.log_retention_days = getattr(settings, 'AI_AUDIT_RETENTION_DAYS', 2555)  # 7 years

    def log_ai_decision(self, decision_type: str, input_data: Dict, output_data: Dict,
                       user_id: Optional[int] = None, model_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Log an AI decision for audit purposes"""
        try:
            log_entry = {
                'log_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'decision_type': decision_type,
                'user_id': user_id,
                'input_data_hash': self._hash_data(input_data),
                'output_data_hash': self._hash_data(output_data),
                'input_data_summary': self._create_data_summary(input_data),
                'output_data_summary': self._create_data_summary(output_data),
                'model_info': model_info or {},
                'compliance_flags': self._check_compliance_flags(input_data, output_data),
                'retention_date': (datetime.now() + timedelta(days=self.log_retention_days)).isoformat()
            }

            # Store log entry (in production, this would go to a secure database)
            cache_key = f"audit_log_{log_entry['log_id']}"
            cache.set(cache_key, log_entry, timeout=self.log_retention_days * 24 * 3600)

            logger.info(f"AI decision logged: {decision_type} for user {user_id}")

            return log_entry

        except Exception as e:
            logger.error(f"AI decision logging failed: {str(e)}")
            return {'error': str(e), 'logging_failed': True}

    def _hash_data(self, data: Dict) -> str:
        """Create hash of data for integrity checking"""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception:
            return 'hash_failed'

    def _create_data_summary(self, data: Dict) -> Dict[str, Any]:
        """Create summary of data for logging"""
        try:
            summary = {
                'data_type': type(data).__name__,
                'size': len(str(data)),
                'keys': list(data.keys()) if isinstance(data, dict) else [],
                'has_sensitive_data': self._check_sensitive_data(data)
            }

            # Add specific summaries based on data type
            if 'case_id' in data:
                summary['case_id'] = data['case_id']

            if 'confidence' in data:
                summary['confidence'] = data['confidence']

            if 'predicted_outcome' in data:
                summary['predicted_outcome'] = data['predicted_outcome']

            return summary

        except Exception as e:
            logger.error(f"Data summary creation failed: {str(e)}")
            return {'error': str(e)}

    def _check_sensitive_data(self, data: Dict) -> bool:
        """Check if data contains sensitive information"""
        try:
            sensitive_patterns = [
                'aadhaar', 'pan', 'passport', 'ssn', 'social_security',
                'bank_account', 'credit_card', 'password', 'secret'
            ]

            data_str = str(data).lower()
            return any(pattern in data_str for pattern in sensitive_patterns)

        except Exception:
            return False

    def _check_compliance_flags(self, input_data: Dict, output_data: Dict) -> List[str]:
        """Check for compliance issues"""
        flags = []

        try:
            # Check for unusual confidence levels
            if 'confidence' in output_data:
                confidence = output_data['confidence']
                if confidence > 0.99:
                    flags.append('unusually_high_confidence')
                elif confidence < 0.1:
                    flags.append('unusually_low_confidence')

            # Check for missing required fields
            required_fields = ['case_id', 'predicted_outcome']
            for field in required_fields:
                if field not in output_data:
                    flags.append(f'missing_required_field_{field}')

            # Check for data processing errors
            if 'error' in output_data:
                flags.append('processing_error_occurred')

            return flags

        except Exception as e:
            logger.error(f"Compliance check failed: {str(e)}")
            return ['compliance_check_failed']

    def get_audit_trail(self, decision_type: Optional[str] = None, user_id: Optional[int] = None,
                       start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict]:
        """Retrieve audit trail with filters"""
        try:
            # In production, this would query a proper audit database
            # For now, return empty list as placeholder
            logger.info(f"Audit trail requested: type={decision_type}, user={user_id}")

            return []

        except Exception as e:
            logger.error(f"Audit trail retrieval failed: {str(e)}")
            return []


class EthicalComplianceChecker:
    """Ensures AI systems comply with ethical guidelines"""

    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        self.bias_detector = BiasDetectionEngine()
        self.audit_logger = AuditLogManager()

    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load ethical compliance rules"""
        return {
            'transparency': {
                'requires_explanation': True,
                'confidence_threshold': 0.6,
                'human_review_threshold': 0.4
            },
            'fairness': {
                'bias_detection_required': True,
                'demographic_parity_threshold': 0.1,
                'equal_opportunity_threshold': 0.1
            },
            'accountability': {
                'audit_logging_required': True,
                'human_oversight_required': True,
                'appeal_mechanism_required': True
            },
            'privacy': {
                'data_minimization': True,
                'anonymization_required': True,
                'retention_limits': True
            }
        }

    def check_compliance(self, ai_output: Dict, input_data: Dict, user_context: Dict) -> Dict[str, Any]:
        """Check if AI output complies with ethical guidelines"""
        try:
            compliance_results = {
                'overall_compliance': True,
                'compliance_score': 1.0,
                'violations': [],
                'recommendations': [],
                'human_review_required': False,
                'check_timestamp': datetime.now().isoformat()
            }

            # Check transparency compliance
            transparency_check = self._check_transparency_compliance(ai_output)
            compliance_results['transparency'] = transparency_check
            if not transparency_check['compliant']:
                compliance_results['overall_compliance'] = False
                compliance_results['violations'].extend(transparency_check['violations'])

            # Check fairness compliance
            fairness_check = self._check_fairness_compliance(ai_output, input_data)
            compliance_results['fairness'] = fairness_check
            if not fairness_check['compliant']:
                compliance_results['overall_compliance'] = False
                compliance_results['violations'].extend(fairness_check['violations'])

            # Check accountability compliance
            accountability_check = self._check_accountability_compliance(ai_output, user_context)
            compliance_results['accountability'] = accountability_check
            if not accountability_check['compliant']:
                compliance_results['overall_compliance'] = False
                compliance_results['violations'].extend(accountability_check['violations'])

            # Check privacy compliance
            privacy_check = self._check_privacy_compliance(input_data, ai_output)
            compliance_results['privacy'] = privacy_check
            if not privacy_check['compliant']:
                compliance_results['overall_compliance'] = False
                compliance_results['violations'].extend(privacy_check['violations'])

            # Calculate overall compliance score
            total_checks = 4  # transparency, fairness, accountability, privacy
            compliant_checks = sum([
                transparency_check['compliant'],
                fairness_check['compliant'],
                accountability_check['compliant'],
                privacy_check['compliant']
            ])
            compliance_results['compliance_score'] = compliant_checks / total_checks

            # Determine if human review is required
            compliance_results['human_review_required'] = (
                not compliance_results['overall_compliance'] or
                compliance_results['compliance_score'] < 0.8 or
                ai_output.get('confidence', 1.0) < 0.6
            )

            # Generate recommendations
            compliance_results['recommendations'] = self._generate_compliance_recommendations(
                compliance_results
            )

            # Log compliance check
            self.audit_logger.log_ai_decision(
                'compliance_check',
                {'ai_output': ai_output, 'input_data': input_data},
                compliance_results,
                user_context.get('user_id')
            )

            return compliance_results

        except Exception as e:
            logger.error(f"Compliance check failed: {str(e)}")
            return {
                'overall_compliance': False,
                'compliance_score': 0.0,
                'error': str(e),
                'violations': ['compliance_check_failed']
            }

    def _check_transparency_compliance(self, ai_output: Dict) -> Dict[str, Any]:
        """Check transparency compliance"""
        violations = []

        # Check if explanation is provided
        if 'explanation' not in ai_output and 'reasoning' not in ai_output:
            violations.append('missing_explanation')

        # Check confidence level
        confidence = ai_output.get('confidence', 0)
        if confidence < self.compliance_rules['transparency']['confidence_threshold']:
            violations.append('low_confidence_without_explanation')

        # Check if limitations are disclosed
        if 'limitations' not in ai_output and 'caveats' not in ai_output:
            violations.append('missing_limitations_disclosure')

        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'score': 1.0 - (len(violations) * 0.3)
        }

    def _check_fairness_compliance(self, ai_output: Dict, input_data: Dict) -> Dict[str, Any]:
        """Check fairness compliance"""
        violations = []

        # Check for bias indicators
        if ai_output.get('bias_detected', False):
            violations.append('bias_detected_in_prediction')

        # Check demographic representation
        if 'demographic_analysis' not in ai_output:
            violations.append('missing_demographic_analysis')

        # Check for equal treatment indicators
        if 'fairness_metrics' not in ai_output:
            violations.append('missing_fairness_metrics')

        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'score': 1.0 - (len(violations) * 0.25)
        }

    def _check_accountability_compliance(self, ai_output: Dict, user_context: Dict) -> Dict[str, Any]:
        """Check accountability compliance"""
        violations = []

        # Check if model information is provided
        if 'model_version' not in ai_output and 'model_info' not in ai_output:
            violations.append('missing_model_information')

        # Check if audit trail is available
        if 'audit_id' not in ai_output:
            violations.append('missing_audit_information')

        # Check if human review options are available
        if 'human_review_required' not in ai_output:
            violations.append('missing_human_review_option')

        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'score': 1.0 - (len(violations) * 0.2)
        }

    def _check_privacy_compliance(self, input_data: Dict, ai_output: Dict) -> Dict[str, Any]:
        """Check privacy compliance"""
        violations = []

        # Check for sensitive data exposure
        if self._contains_sensitive_data(ai_output):
            violations.append('sensitive_data_exposed')

        # Check for data minimization
        if len(str(ai_output)) > 10000:  # Arbitrary threshold
            violations.append('excessive_data_disclosure')

        # Check for anonymization
        if 'personal_identifiers' in ai_output:
            violations.append('personal_data_not_anonymized')

        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'score': 1.0 - (len(violations) * 0.25)
        }

    def _contains_sensitive_data(self, data: Dict) -> bool:
        """Check if data contains sensitive information"""
        sensitive_patterns = [
            r'\b\d{4}\s*-\s*\d{4}\s*-\s*\d{4}\b',  # Credit card pattern
            r'\b\d{12}\b',  # Aadhaar pattern
            r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN pattern
        ]

        data_str = str(data)
        for pattern in sensitive_patterns:
            if re.search(pattern, data_str):
                return True

        return False

    def _generate_compliance_recommendations(self, compliance_results: Dict) -> List[str]:
        """Generate recommendations based on compliance check results"""
        recommendations = []

        try:
            if not compliance_results.get('transparency', {}).get('compliant', True):
                recommendations.append("Add detailed explanations and confidence disclosures")

            if not compliance_results.get('fairness', {}).get('compliant', True):
                recommendations.append("Implement bias detection and mitigation measures")

            if not compliance_results.get('accountability', {}).get('compliant', True):
                recommendations.append("Ensure proper audit logging and model transparency")

            if not compliance_results.get('privacy', {}).get('compliant', True):
                recommendations.append("Review data handling and anonymization practices")

            if compliance_results.get('human_review_required', False):
                recommendations.append("Human review recommended before action")

            if not recommendations:
                recommendations.append("All compliance checks passed - continue monitoring")

        except Exception as e:
            logger.error(f"Compliance recommendations generation failed: {str(e)}")

        return recommendations


# Global instances
bias_detector = BiasDetectionEngine()
explainability_engine = ExplainabilityEngine()
audit_logger = AuditLogManager()
ethical_compliance_checker = EthicalComplianceChecker()


def ensure_ethical_ai(ai_output: Dict, input_data: Dict, user_context: Dict,
                     case_context: Optional[Dict] = None) -> Dict[str, Any]:
    """Main function to ensure ethical AI practices"""
    try:
        # Generate explanation
        explanation = explainability_engine.generate_prediction_explanation(
            ai_output, case_context or {}, ai_output.get('model_info', {})
        )

        # Check compliance
        compliance_check = ethical_compliance_checker.check_compliance(
            ai_output, input_data, user_context
        )

        # Check for bias if demographic data is available
        bias_check = {}
        if 'demographic_data' in user_context:
            bias_check = bias_detector.detect_prediction_bias(
                [ai_output], [user_context['demographic_data']]
            )

        # Combine results
        ethical_ai_result = {
            'original_prediction': ai_output,
            'explanation': explanation,
            'compliance_check': compliance_check,
            'bias_analysis': bias_check,
            'ethical_guidelines_met': compliance_check.get('overall_compliance', False),
            'human_review_required': (
                compliance_check.get('human_review_required', False) or
                explanation.get('human_review_required', {}).get('review_required', False)
            ),
            'transparency_score': explanation.get('explanation_quality', {}).get('overall_quality_score', 0),
            'ethical_assessment_date': datetime.now().isoformat()
        }

        # Log the ethical AI assessment
        audit_logger.log_ai_decision(
            'ethical_ai_assessment',
            {'ai_output': ai_output, 'input_data': input_data},
            ethical_ai_result,
            user_context.get('user_id')
        )

        return ethical_ai_result

    except Exception as e:
        logger.error(f"Ethical AI assessment failed: {str(e)}")
        return {
            'error': str(e),
            'ethical_guidelines_met': False,
            'human_review_required': True,
            'original_prediction': ai_output
        }