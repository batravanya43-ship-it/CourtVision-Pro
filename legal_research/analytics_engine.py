"""
Predictive Analytics Engine for CourtVision Pro
Real-time analytics and predictive insights for commercial courts
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from django.db.models import Q, Count, Avg, Sum, StdDev
from django.db.models.functions import TruncDate, TruncMonth, ExtractYear
from django.utils import timezone
from django.core.cache import cache

from .models import Case, HighCourt, AnalyticsData, SearchHistory, UserProfile
from .ai_integration import predictive_analytics
from .ml_models import case_outcome_predictor, legal_trend_analyzer

logger = logging.getLogger(__name__)


class AnalyticsEngineError(Exception):
    """Custom exception for analytics engine errors"""
    pass


class CaseOutcomeAnalytics:
    """Analytics for case outcomes and predictions"""

    def __init__(self):
        self.prediction_cache = {}
        self.cache_timeout = 3600  # 1 hour

    async def predict_case_outcomes_batch(self, cases: List[Case]) -> List[Dict[str, Any]]:
        """Predict outcomes for multiple cases"""
        results = []

        for case in cases:
            try:
                # Check cache first
                cache_key = f"outcome_prediction_{case.id}"
                cached_prediction = cache.get(cache_key)
                if cached_prediction:
                    results.append(cached_prediction)
                    continue

                # Get historical data for prediction
                historical_data = self._get_historical_cases_for_prediction(case)

                # Make prediction
                prediction = await predictive_analytics.predict_case_outcome(case, historical_data)

                # Add case context
                prediction.update({
                    'case_id': str(case.id),
                    'case_title': case.title,
                    'court': case.court.name,
                    'case_type': case.case_type,
                    'prediction_date': datetime.now().isoformat()
                })

                # Cache result
                cache.set(cache_key, prediction, timeout=self.cache_timeout)
                results.append(prediction)

            except Exception as e:
                logger.error(f"Failed to predict outcome for case {case.id}: {str(e)}")
                results.append({
                    'case_id': str(case.id),
                    'case_title': case.title,
                    'error': str(e),
                    'prediction_failed': True
                })

        return results

    def _get_historical_cases_for_prediction(self, case: Case, limit: int = 50) -> List[Dict]:
        """Get historical cases for prediction context"""
        try:
            # Find similar cases
            similar_cases = Case.objects.filter(
                court=case.court,
                case_type=case.case_type,
                judgment_date__gte=case.judgment_date - timedelta(days=365*5)
            ).exclude(id=case.id)[:limit]

            historical_data = []
            for hist_case in similar_cases:
                outcome = self._determine_case_outcome(hist_case)
                if outcome:
                    historical_data.append({
                        'title': hist_case.title,
                        'outcome': outcome,
                        'judgment_date': hist_case.judgment_date.isoformat(),
                        'duration': (hist_case.decision_date - hist_case.judgment_date).days,
                        'court': hist_case.court.name,
                        'tags': [tag.name for tag in hist_case.tags.all()]
                    })

            return historical_data

        except Exception as e:
            logger.error(f"Failed to get historical cases: {str(e)}")
            return []

    def _determine_case_outcome(self, case: Case) -> Optional[str]:
        """Determine case outcome from available data"""
        if case.ai_summary and isinstance(case.ai_summary, dict):
            decision = case.ai_summary.get('decision', '').lower()
            if 'allowed' in decision or 'granted' in decision:
                return 'petitioner_favorable'
            elif 'dismissed' in decision or 'rejected' in decision:
                return 'respondent_favorable'
            elif 'partially' in decision:
                return 'partial'

        return 'unknown'

    def generate_outcome_predictions_dashboard(self, time_period: int = 90) -> Dict[str, Any]:
        """Generate comprehensive outcome predictions dashboard"""
        try:
            cutoff_date = timezone.now() - timedelta(days=time_period)
            recent_cases = Case.objects.filter(
                judgment_date__gte=cutoff_date,
                is_published=True
            ).select_related('court').prefetch_related('tags')

            if not recent_cases.exists():
                return self._empty_outcome_dashboard()

            # Get predictions for recent cases
            predictions = asyncio.run(self.predict_case_outcomes_batch(list(recent_cases[:100])))

            # Analyze predictions
            prediction_stats = self._analyze_predictions(predictions)

            # Generate visualizations
            charts = self._generate_prediction_charts(predictions)

            return {
                'summary': prediction_stats,
                'charts': charts,
                'recent_predictions': predictions[:20],
                'analytics_metadata': {
                    'time_period_days': time_period,
                    'total_cases_analyzed': len(predictions),
                    'analysis_date': datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Failed to generate outcome predictions dashboard: {str(e)}")
            return self._empty_outcome_dashboard()

    def _analyze_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Analyze prediction results"""
        valid_predictions = [p for p in predictions if not p.get('prediction_failed')]

        if not valid_predictions:
            return {'error': 'No valid predictions available'}

        # Count predicted outcomes
        outcome_counts = Counter([p.get('predicted_outcome', 'unknown') for p in valid_predictions])

        # Calculate confidence statistics
        confidences = [p.get('confidence', 0) for p in valid_predictions if p.get('confidence') is not None]
        avg_confidence = statistics.mean(confidences) if confidences else 0
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0

        # Analyze by court
        court_predictions = defaultdict(list)
        for p in valid_predictions:
            court = p.get('court', 'Unknown')
            court_predictions[court].append(p)

        court_stats = {}
        for court, court_preds in court_predictions.items():
            court_confidences = [p.get('confidence', 0) for p in court_preds if p.get('confidence') is not None]
            court_stats[court] = {
                'case_count': len(court_preds),
                'avg_confidence': statistics.mean(court_confidences) if court_confidences else 0,
                'common_outcome': Counter([p.get('predicted_outcome', 'unknown') for p in court_preds]).most_common(1)[0][0] if court_preds else 'unknown'
            }

        return {
            'total_predictions': len(valid_predictions),
            'successful_predictions': len(valid_predictions),
            'failed_predictions': len(predictions) - len(valid_predictions),
            'outcome_distribution': dict(outcome_counts),
            'average_confidence': round(avg_confidence, 3),
            'confidence_std': round(confidence_std, 3),
            'court_analysis': court_stats,
            'high_confidence_predictions': len([c for c in confidences if c > 0.8]),
            'low_confidence_predictions': len([c for c in confidences if c < 0.5])
        }

    def _generate_prediction_charts(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Generate charts for predictions"""
        charts = {}

        try:
            valid_predictions = [p for p in predictions if not p.get('prediction_failed')]

            if not valid_predictions:
                return charts

            # Outcome distribution pie chart
            outcomes = [p.get('predicted_outcome', 'unknown') for p in valid_predictions]
            outcome_counts = Counter(outcomes)

            fig_pie = go.Figure(data=[go.Pie(
                labels=list(outcome_counts.keys()),
                values=list(outcome_counts.values()),
                title="Predicted Case Outcomes"
            )])
            charts['outcome_distribution'] = json.dumps(fig_pie, cls=PlotlyJSONEncoder)

            # Confidence distribution histogram
            confidences = [p.get('confidence', 0) for p in valid_predictions if p.get('confidence') is not None]
            if confidences:
                fig_hist = go.Figure(data=[go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    title="Prediction Confidence Distribution"
                )])
                fig_hist.update_xaxis(title="Confidence Score")
                fig_hist.update_yaxis(title="Number of Cases")
                charts['confidence_distribution'] = json.dumps(fig_hist, cls=PlotlyJSONEncoder)

            # Court-wise predictions
            court_data = defaultdict(list)
            for p in valid_predictions:
                court = p.get('court', 'Unknown')
                confidence = p.get('confidence', 0)
                if confidence:
                    court_data[court].append(confidence)

            if court_data:
                courts = list(court_data.keys())
                avg_confidences = [statistics.mean(court_data[c]) for c in courts]
                case_counts = [len(court_data[c]) for c in courts]

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=courts,
                    y=avg_confidences,
                    name='Average Confidence',
                    yaxis='y'
                ))
                fig_bar.add_trace(go.Scatter(
                    x=courts,
                    y=case_counts,
                    mode='markers',
                    name='Case Count',
                    yaxis='y2'
                ))

                fig_bar.update_layout(
                    title="Court-wise Prediction Analysis",
                    yaxis=dict(title="Average Confidence"),
                    yaxis2=dict(title="Case Count", overlaying='y', side='right')
                )
                charts['court_analysis'] = json.dumps(fig_bar, cls=PlotlyJSONEncoder)

        except Exception as e:
            logger.error(f"Failed to generate charts: {str(e)}")

        return charts

    def _empty_outcome_dashboard(self) -> Dict[str, Any]:
        """Return empty dashboard structure"""
        return {
            'summary': {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'outcome_distribution': {},
                'average_confidence': 0,
                'court_analysis': {}
            },
            'charts': {},
            'recent_predictions': [],
            'analytics_metadata': {
                'time_period_days': 0,
                'total_cases_analyzed': 0,
                'analysis_date': datetime.now().isoformat(),
                'status': 'no_data'
            }
        }


class JudicialPatternAnalytics:
    """Analytics for judicial decision patterns"""

    def __init__(self):
        self.pattern_cache = {}
        self.cache_timeout = 7200  # 2 hours

    def analyze_judicial_patterns(self, court_id: Optional[int] = None,
                                  time_period: int = 365) -> Dict[str, Any]:
        """Analyze judicial decision patterns"""
        cache_key = f"judicial_patterns_{court_id or 'all'}_{time_period}"

        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            cutoff_date = timezone.now() - timedelta(days=time_period)

            # Query cases
            cases_query = Case.objects.filter(judgment_date__gte=cutoff_date)
            if court_id:
                cases_query = cases_query.filter(court_id=court_id)

            cases = cases_query.select_related('court').prefetch_related('tags')

            if not cases.exists():
                return self._empty_pattern_analysis()

            # Analyze different patterns
            patterns = {
                'decision_timing': self._analyze_decision_timing(cases),
                'case_type_patterns': self._analyze_case_type_patterns(cases),
                'tag_patterns': self._analyze_tag_patterns(cases),
                'outcome_patterns': self._analyze_outcome_patterns(cases),
                'seasonal_patterns': self._analyze_seasonal_patterns(cases),
                'complexity_patterns': self._analyze_complexity_patterns(cases)
            }

            # Generate insights
            insights = self._generate_judicial_insights(patterns)

            result = {
                'patterns': patterns,
                'insights': insights,
                'metadata': {
                    'court_id': court_id,
                    'time_period_days': time_period,
                    'total_cases': cases.count(),
                    'analysis_date': datetime.now().isoformat()
                }
            }

            # Cache result
            cache.set(cache_key, result, timeout=self.cache_timeout)

            return result

        except Exception as e:
            logger.error(f"Judicial pattern analysis failed: {str(e)}")
            return self._empty_pattern_analysis()

    def _analyze_decision_timing(self, cases) -> Dict[str, Any]:
        """Analyze timing of decisions"""
        durations = []
        for case in cases:
            duration = (case.decision_date - case.judgment_date).days
            durations.append(duration)

        if not durations:
            return {}

        return {
            'average_duration_days': round(statistics.mean(durations), 1),
            'median_duration_days': round(statistics.median(durations), 1),
            'min_duration_days': min(durations),
            'max_duration_days': max(durations),
            'std_deviation': round(statistics.stdev(durations) if len(durations) > 1 else 0, 1),
            'fast_decisions': len([d for d in durations if d <= 30]),
            'slow_decisions': len([d for d in durations if d > 180])
        }

    def _analyze_case_type_patterns(self, cases) -> Dict[str, Any]:
        """Analyze patterns by case type"""
        case_type_counts = cases.values('case_type').annotate(
            count=Count('id'),
            avg_duration=Avg((ExtractYear('decision_date') - ExtractYear('judgment_date')) * 365 +
                           (ExtractYear('decision_date') - ExtractYear('judgment_date')))
        )

        patterns = {}
        total_cases = sum(item['count'] for item in case_type_counts)

        for item in case_type_counts:
            case_type = item['case_type']
            percentage = (item['count'] / total_cases) * 100 if total_cases > 0 else 0

            patterns[case_type] = {
                'count': item['count'],
                'percentage': round(percentage, 2),
                'avg_duration_days': round(item['avg_duration'] or 0, 1),
                'frequency': 'high' if percentage > 30 else 'medium' if percentage > 10 else 'low'
            }

        return patterns

    def _analyze_tag_patterns(self, cases) -> Dict[str, Any]:
        """Analyze patterns in legal topics/tags"""
        tag_counts = defaultdict(int)
        tag_outcomes = defaultdict(lambda: defaultdict(int))

        for case in cases:
            for tag in case.tags.all():
                tag_name = tag.name
                tag_counts[tag_name] += 1

                # Associate with outcome
                outcome = self._determine_case_outcome(case)
                if outcome:
                    tag_outcomes[tag_name][outcome] += 1

        # Sort by frequency
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

        patterns = {}
        for tag_name, count in sorted_tags[:20]:  # Top 20 tags
            outcomes = tag_outcomes.get(tag_name, {})
            total_outcomes = sum(outcomes.values())
            most_common_outcome = max(outcomes.items(), key=lambda x: x[1])[0] if outcomes else 'unknown'

            patterns[tag_name] = {
                'frequency': count,
                'most_common_outcome': most_common_outcome,
                'outcome_distribution': dict(outcomes),
                'success_rate': (outcomes.get('petitioner_favorable', 0) / total_outcomes * 100) if total_outcomes > 0 else 0
            }

        return patterns

    def _analyze_outcome_patterns(self, cases) -> Dict[str, Any]:
        """Analyze outcome patterns"""
        outcomes = []
        for case in cases:
            outcome = self._determine_case_outcome(case)
            if outcome:
                outcomes.append(outcome)

        if not outcomes:
            return {}

        outcome_counts = Counter(outcomes)
        total_cases = len(outcomes)

        patterns = {}
        for outcome, count in outcome_counts.items():
            patterns[outcome] = {
                'count': count,
                'percentage': round((count / total_cases) * 100, 2),
                'trend': 'stable'  # Would compare with historical data
            }

        return patterns

    def _analyze_seasonal_patterns(self, cases) -> Dict[str, Any]:
        """Analyze seasonal patterns in judgments"""
        monthly_counts = defaultdict(int)

        for case in cases:
            month = case.judgment_date.month
            monthly_counts[month] += 1

        if not monthly_counts:
            return {}

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly_data = []
        for i, month_name in enumerate(month_names, 1):
            count = monthly_counts.get(i, 0)
            monthly_data.append({
                'month': month_name,
                'count': count,
                'percentage': round((count / len(cases)) * 100, 2) if cases else 0
            })

        # Find peak and low months
        peak_month = max(monthly_data, key=lambda x: x['count'])
        low_month = min(monthly_data, key=lambda x: x['count'])

        return {
            'monthly_distribution': monthly_data,
            'peak_month': peak_month,
            'lowest_month': low_month,
            'seasonal_variance': round(statistics.stdev([m['count'] for m in monthly_data]) if len(monthly_data) > 1 else 0, 1)
        }

    def _analyze_complexity_patterns(self, cases) -> Dict[str, Any]:
        """Analyze case complexity patterns"""
        complexity_metrics = []

        for case in cases:
            # Simple complexity metrics
            text_length = len(case.case_text)
            tag_count = case.tags.count()
            precedent_count = len(case.precedents_cited) if case.precedents_cited else 0
            statute_count = len(case.statutes_cited) if case.statutes_cited else 0

            # Calculate complexity score
            complexity_score = (
                min(text_length / 10000, 1) * 0.4 +  # Text length (max 10k chars)
                min(tag_count / 10, 1) * 0.2 +        # Tag count (max 10)
                min(precedent_count / 20, 1) * 0.2 +  # Precedent count (max 20)
                min(statute_count / 10, 1) * 0.2      # Statute count (max 10)
            )

            complexity_metrics.append({
                'case_id': str(case.id),
                'complexity_score': complexity_score,
                'text_length': text_length,
                'tag_count': tag_count,
                'precedent_count': precedent_count,
                'statute_count': statute_count,
                'duration_days': (case.decision_date - case.judgment_date).days
            })

        if not complexity_metrics:
            return {}

        # Analyze complexity vs duration
        complexities = [m['complexity_score'] for m in complexity_metrics]
        durations = [m['duration_days'] for m in complexity_metrics]

        # Simple correlation calculation
        if len(complexities) > 1 and len(set(complexities)) > 1:
            correlation = np.corrcoef(complexities, durations)[0, 1]
        else:
            correlation = 0

        return {
            'average_complexity': round(statistics.mean(complexities), 3),
            'complexity_distribution': {
                'low': len([c for c in complexities if c < 0.3]),
                'medium': len([c for c in complexities if 0.3 <= c < 0.7]),
                'high': len([c for c in complexities if c >= 0.7])
            },
            'complexity_duration_correlation': round(correlation, 3),
            'most_complex_cases': sorted(complexity_metrics, key=lambda x: x['complexity_score'], reverse=True)[:5]
        }

    def _determine_case_outcome(self, case: Case) -> Optional[str]:
        """Determine case outcome"""
        if case.ai_summary and isinstance(case.ai_summary, dict):
            decision = case.ai_summary.get('decision', '').lower()
            if 'allowed' in decision or 'granted' in decision:
                return 'petitioner_favorable'
            elif 'dismissed' in decision or 'rejected' in decision:
                return 'respondent_favorable'
            elif 'partially' in decision:
                return 'partial'

        return 'unknown'

    def _generate_judicial_insights(self, patterns: Dict) -> List[str]:
        """Generate insights from pattern analysis"""
        insights = []

        try:
            # Decision timing insights
            timing = patterns.get('decision_timing', {})
            if timing.get('average_duration_days', 0) > 90:
                insights.append("Cases take longer than average to resolve (over 90 days)")

            if timing.get('fast_decisions', 0) > timing.get('slow_decisions', 0):
                insights.append("Majority of cases are resolved quickly (within 30 days)")

            # Case type insights
            case_types = patterns.get('case_type_patterns', {})
            if case_types:
                most_common = max(case_types.items(), key=lambda x: x[1]['count'])
                insights.append(f"Most common case type: {most_common[0]} ({most_common[1]['percentage']}%)")

            # Complexity insights
            complexity = patterns.get('complexity_patterns', {})
            if complexity.get('complexity_duration_correlation', 0) > 0.5:
                insights.append("Strong correlation between case complexity and resolution time")

            # Seasonal insights
            seasonal = patterns.get('seasonal_patterns', {})
            if seasonal:
                peak = seasonal.get('peak_month', {})
                insights.append(f"Peak judgment month: {peak.get('month', 'Unknown')}")

        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")

        return insights

    def _empty_pattern_analysis(self) -> Dict[str, Any]:
        """Return empty pattern analysis structure"""
        return {
            'patterns': {},
            'insights': [],
            'metadata': {
                'court_id': None,
                'time_period_days': 0,
                'total_cases': 0,
                'analysis_date': datetime.now().isoformat(),
                'status': 'no_data'
            }
        }


class RiskAssessmentAnalytics:
    """Risk assessment for commercial disputes"""

    def __init__(self):
        self.risk_factors = {
            'case_complexity': 0.3,
            'jurisdiction_backlog': 0.2,
            'legal_precedent_clarity': 0.2,
            'dispute_amount': 0.15,
            'parties_cooperation': 0.15
        }

    def assess_case_risk(self, case: Case) -> Dict[str, Any]:
        """Assess risk for a specific case"""
        try:
            risk_scores = {}

            # Case complexity risk
            risk_scores['complexity'] = self._assess_complexity_risk(case)

            # Jurisdiction backlog risk
            risk_scores['jurisdiction'] = self._assess_jurisdiction_risk(case.court)

            # Precedent clarity risk
            risk_scores['precedent'] = self._assess_precedent_risk(case)

            # Overall risk score
            overall_risk = sum(
                score * self.risk_factors[factor]
                for factor, score in risk_scores.items()
                if factor in self.risk_factors
            )

            risk_level = self._categorize_risk(overall_risk)

            # Generate recommendations
            recommendations = self._generate_risk_recommendations(risk_scores, overall_risk)

            return {
                'case_id': str(case.id),
                'case_title': case.title,
                'overall_risk_score': round(overall_risk, 3),
                'risk_level': risk_level,
                'risk_factors': risk_scores,
                'recommendations': recommendations,
                'assessment_date': datetime.now().isoformat(),
                'risk_factors_used': list(self.risk_factors.keys())
            }

        except Exception as e:
            logger.error(f"Risk assessment failed for case {case.id}: {str(e)}")
            return {
                'case_id': str(case.id),
                'error': str(e),
                'assessment_failed': True
            }

    def _assess_complexity_risk(self, case: Case) -> float:
        """Assess risk based on case complexity"""
        complexity_score = 0

        # Text length complexity
        text_length = len(case.case_text)
        if text_length > 20000:
            complexity_score += 0.4
        elif text_length > 10000:
            complexity_score += 0.2

        # Number of issues
        tag_count = case.tags.count()
        if tag_count > 10:
            complexity_score += 0.3
        elif tag_count > 5:
            complexity_score += 0.15

        # Legal citations
        precedent_count = len(case.precedents_cited) if case.precedents_cited else 0
        statute_count = len(case.statutes_cited) if case.statutes_cited else 0
        if precedent_count > 20 or statute_count > 15:
            complexity_score += 0.3
        elif precedent_count > 10 or statute_count > 8:
            complexity_score += 0.15

        return min(complexity_score, 1.0)

    def _assess_jurisdiction_risk(self, court: HighCourt) -> float:
        """Assess risk based on jurisdiction factors"""
        # Simplified risk assessment based on court characteristics
        risk_score = 0.2  # Base risk

        # Could be enhanced with real data about court backlogs, etc.
        court_name = court.name.lower()

        # Higher risk for larger, busier courts (simplified heuristic)
        if 'bombay' in court_name or 'delhi' in court_name or 'calcutta' in court_name:
            risk_score += 0.3

        # Lower risk for specialized commercial courts
        if 'commercial' in court_name:
            risk_score -= 0.2

        return max(0.0, min(risk_score, 1.0))

    def _assess_precedent_risk(self, case: Case) -> float:
        """Assess risk based on precedent clarity"""
        risk_score = 0.3  # Base risk

        # More precedents = clearer legal position = lower risk
        precedent_count = len(case.precedents_cited) if case.precedents_cited else 0
        if precedent_count > 15:
            risk_score -= 0.3
        elif precedent_count > 8:
            risk_score -= 0.15
        elif precedent_count < 3:
            risk_score += 0.3

        # Recent precedents are more reliable
        # This would need more sophisticated analysis in practice

        return max(0.0, min(risk_score, 1.0))

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score >= 0.7:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def _generate_risk_recommendations(self, risk_scores: Dict[str, float], overall_risk: float) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []

        if risk_scores.get('complexity', 0) > 0.6:
            recommendations.append("Consider breaking down complex issues into separate proceedings")
            recommendations.append("Prepare comprehensive documentation to reduce ambiguity")

        if risk_scores.get('jurisdiction', 0) > 0.6:
            recommendations.append("Consider alternative dispute resolution mechanisms")
            recommendations.append("Allow extra time for procedural requirements")

        if risk_scores.get('precedent', 0) > 0.6:
            recommendations.append("Invest in additional legal research to strengthen case")
            recommendations.append("Consider expert testimony on novel legal issues")

        if overall_risk > 0.7:
            recommendations.append("High-risk case - consider settlement negotiations")
            recommendations.append("Prepare for extended timeline and increased costs")

        if not recommendations:
            recommendations.append("Standard risk management practices should be sufficient")

        return recommendations

    def batch_risk_assessment(self, cases: List[Case]) -> List[Dict[str, Any]]:
        """Perform risk assessment for multiple cases"""
        results = []

        for case in cases:
            risk_assessment = self.assess_case_risk(case)
            results.append(risk_assessment)

        return results


# Global analytics instances
case_outcome_analytics = CaseOutcomeAnalytics()
judicial_pattern_analytics = JudicialPatternAnalytics()
risk_assessment_analytics = RiskAssessmentAnalytics()


async def generate_comprehensive_analytics(time_period: int = 90, court_id: Optional[int] = None) -> Dict[str, Any]:
    """Generate comprehensive analytics dashboard"""
    try:
        # Get recent cases
        cutoff_date = timezone.now() - timedelta(days=time_period)
        cases = Case.objects.filter(
            judgment_date__gte=cutoff_date,
            is_published=True
        ).select_related('court').prefetch_related('tags')

        if not cases.exists():
            return {'error': 'No data available for analysis period'}

        # Generate different analytics components
        outcome_predictions = case_outcome_analytics.generate_outcome_predictions_dashboard(time_period)
        judicial_patterns = judicial_pattern_analytics.analyze_judicial_patterns(court_id, time_period)

        # Risk assessment for recent cases
        risk_assessments = risk_assessment_analytics.batch_risk_assessment(list(cases[:50]))

        # Summary statistics
        summary_stats = {
            'total_cases': cases.count(),
            'courts_represented': cases.values('court__name').distinct().count(),
            'case_types': list(cases.values_list('case_type', flat=True).distinct()),
            'average_duration': cases.annotate(
                duration=(ExtractYear('decision_date') - ExtractYear('judgment_date')) * 365 +
                         (ExtractYear('decision_date') - ExtractYear('judgment_date'))
            ).aggregate(avg_duration=Avg('duration'))['avg_duration'] or 0
        }

        return {
            'summary_statistics': summary_stats,
            'outcome_predictions': outcome_predictions,
            'judicial_patterns': judicial_patterns,
            'risk_assessments': risk_assessments[:20],  # Top 20 risk assessments
            'analytics_metadata': {
                'time_period_days': time_period,
                'court_id': court_id,
                'generation_date': datetime.now().isoformat(),
                'data_freshness': 'real_time'
            }
        }

    except Exception as e:
        logger.error(f"Comprehensive analytics generation failed: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }