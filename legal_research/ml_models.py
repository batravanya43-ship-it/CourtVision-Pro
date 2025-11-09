"""
Machine Learning Models for CourtVision Pro
Implements predictive analytics and classification models for legal data
"""

import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from django.db.models import Q, Count, Avg
from django.utils import timezone

from .models import Case, HighCourt, AnalyticsData, SearchHistory

logger = logging.getLogger(__name__)


class CaseOutcomePredictor:
    """Predicts case outcomes based on historical patterns"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.scaler = None
        self.is_trained = False
        self.model_type = 'random_forest'
        self.feature_columns = []

    def train_model(self, training_data: List[Dict] = None) -> Dict[str, Any]:
        """Train the case outcome prediction model"""
        try:
            if training_data is None:
                training_data = self._prepare_training_data()

            if not training_data:
                return {
                    'success': False,
                    'message': 'No training data available',
                    'accuracy': 0.0
                }

            # Convert to DataFrame
            df = pd.DataFrame(training_data)

            # Feature engineering
            X, y = self._engineer_features(df)

            if len(X) < 10:
                return {
                    'success': False,
                    'message': 'Insufficient training data',
                    'accuracy': 0.0
                }

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )

            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            mean_cv_score = cv_scores.mean()

            self.is_trained = True

            # Save model
            self._save_model()

            logger.info(f"Model trained with accuracy: {accuracy:.3f}, CV score: {mean_cv_score:.3f}")

            return {
                'success': True,
                'accuracy': accuracy,
                'cv_score': mean_cv_score,
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
                'training_samples': len(X),
                'test_samples': len(X_test)
            }

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'accuracy': 0.0
            }

    def predict(self, case_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcome for a single case"""
        if not self.is_trained:
            self._load_model()

        if not self.is_trained:
            return {
                'predicted_outcome': 'Model not trained',
                'confidence': 0.0,
                'probabilities': {}
            }

        try:
            # Convert features to DataFrame
            df = pd.DataFrame([case_features])
            X = self._transform_features(df)

            if X.shape[1] != len(self.feature_columns):
                # Handle missing features
                missing_cols = set(self.feature_columns) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[self.feature_columns]

            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            # Get feature importance for this case
            feature_contributions = dict(zip(X.columns, self.model.feature_importances_))

            return {
                'predicted_outcome': prediction,
                'confidence': max(probabilities),
                'probabilities': dict(zip(self.label_encoder.classes_, probabilities)),
                'feature_contributions': feature_contributions,
                'model_version': 'v1.0'
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'predicted_outcome': 'Prediction failed',
                'confidence': 0.0,
                'probabilities': {},
                'error': str(e)
            }

    def _prepare_training_data(self) -> List[Dict]:
        """Prepare training data from database"""
        try:
            # Get historical cases with outcomes
            cases = Case.objects.filter(
                ai_summary__isnull=False,
                court__isnull=False
            ).select_related('court').prefetch_related('tags')

            training_data = []

            for case in cases:
                # Extract features from case
                features = self._extract_case_features(case)

                # Determine outcome from AI summary or tags
                outcome = self._determine_outcome(case)
                if outcome:
                    features['outcome'] = outcome
                    training_data.append(features)

            return training_data

        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            return []

    def _extract_case_features(self, case: Case) -> Dict[str, Any]:
        """Extract features from a case"""
        features = {
            'court_id': case.court.id,
            'case_type': case.case_type,
            'judgment_year': case.judgment_date.year,
            'judgment_month': case.judgment_date.month,
            'days_to_decision': (case.decision_date - case.judgment_date).days,
            'text_length': len(case.case_text),
            'tag_count': case.tags.count(),
            'has_precedents': len(case.precedents_cited) > 0 if case.precedents_cited else False,
            'statute_count': len(case.statutes_cited) if case.statutes_cited else 0,
            'view_count': case.view_count,
            'relevance_score': case.relevance_score
        }

        # Add tag features
        for tag in case.tags.all():
            features[f'tag_{tag.name.lower().replace(" ", "_")}'] = 1

        return features

    def _determine_outcome(self, case: Case) -> Optional[str]:
        """Determine case outcome from available data"""
        # This is a simplified implementation
        # In practice, you'd parse the judgment text or use AI to determine outcome

        # Check AI summary for outcome information
        if case.ai_summary and isinstance(case.ai_summary, dict):
            decision = case.ai_summary.get('decision', '').lower()
            if 'allowed' in decision or 'granted' in decision:
                return 'petitioner_favorable'
            elif 'dismissed' in decision or 'rejected' in decision:
                return 'respondent_favorable'
            elif 'partially' in decision:
                return 'partial'
            elif 'remanded' in decision:
                return 'remanded'

        # Fallback based on case type and other factors
        return 'unknown'

    def _engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Engineer features from raw data"""
        # Handle categorical variables
        categorical_columns = ['court_id', 'case_type']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Create label encoder for target
        if 'outcome' in df.columns:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(df['outcome'])
        else:
            y = pd.Series([0] * len(df))

        # Drop target and text columns
        X = df.drop(['outcome'], axis=1, errors='ignore')

        # Handle missing values
        X = X.fillna(0)

        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            self.scaler = StandardScaler()
            X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])

        self.feature_columns = list(X.columns)

        return X, y

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders"""
        # Ensure all expected columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Scale numerical features
        if self.scaler:
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 0:
                df[numerical_columns] = self.scaler.transform(df[numerical_columns])

        return df[self.feature_columns]

    def _save_model(self):
        """Save trained model to disk"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type,
                'trained_at': datetime.now().isoformat()
            }

            model_path = 'legal_research/models/case_outcome_predictor.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info("Model saved successfully")

        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")

    def _load_model(self):
        """Load trained model from disk"""
        try:
            model_path = 'legal_research/models/case_outcome_predictor.pkl'
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.is_trained = True

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_trained = False


class LegalTrendAnalyzer:
    """Analyzes emerging legal trends across courts and time"""

    def __init__(self):
        self.trend_cache = {}
        self.cache_timeout = 3600  # 1 hour

    def analyze_trends(self, time_period: int = 365, court_id: Optional[int] = None) -> Dict[str, Any]:
        """Analyze legal trends over specified time period"""
        cache_key = f"trends_{time_period}_{court_id or 'all'}"

        if cache_key in self.trend_cache:
            cache_time, cached_data = self.trend_cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=self.cache_timeout):
                return cached_data

        try:
            cutoff_date = timezone.now() - timedelta(days=time_period)

            # Query cases
            cases_query = Case.objects.filter(judgment_date__gte=cutoff_date)
            if court_id:
                cases_query = cases_query.filter(court_id=court_id)

            cases = cases_query.select_related('court').prefetch_related('tags')

            if not cases.exists():
                return self._empty_trend_analysis()

            # Analyze different trend dimensions
            trends = {
                'case_type_trends': self._analyze_case_type_trends(cases),
                'court_trends': self._analyze_court_trends(cases),
                'tag_trends': self._analyze_tag_trends(cases),
                'temporal_trends': self._analyze_temporal_trends(cases, time_period),
                'outcome_trends': self._analyze_outcome_trends(cases),
                'emerging_topics': self._identify_emerging_topics(cases),
                'analysis_metadata': {
                    'time_period_days': time_period,
                    'total_cases': cases.count(),
                    'courts_analyzed': len(set(cases.values_list('court_id', flat=True))),
                    'analysis_date': datetime.now().isoformat()
                }
            }

            # Cache results
            self.trend_cache[cache_key] = (datetime.now(), trends)

            return trends

        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            return self._empty_trend_analysis()

    def _analyze_case_type_trends(self, cases) -> Dict[str, Any]:
        """Analyze trends in case types"""
        case_type_counts = cases.values('case_type').annotate(count=Count('id'))

        total_cases = sum(item['count'] for item in case_type_counts)

        trends = {}
        for item in case_type_counts:
            percentage = (item['count'] / total_cases) * 100 if total_cases > 0 else 0
            trends[item['case_type']] = {
                'count': item['count'],
                'percentage': round(percentage, 2),
                'trend_direction': 'stable'  # Would compare with historical data
            }

        return trends

    def _analyze_court_trends(self, cases) -> Dict[str, Any]:
        """Analyze trends across different courts"""
        court_counts = cases.values('court__name').annotate(
            count=Count('id'),
            avg_relevance=Avg('relevance_score')
        )

        trends = {}
        for item in court_counts:
            trends[item['court__name']] = {
                'case_count': item['count'],
                'avg_relevance': round(item['avg_relevance'] or 0, 2),
                'court_activity': 'high' if item['count'] > 50 else 'medium' if item['count'] > 20 else 'low'
            }

        return trends

    def _analyze_tag_trends(self, cases) -> Dict[str, Any]:
        """Analyze trending legal topics via tags"""
        from django.db.models import Count

        tag_counts = {}
        for case in cases:
            for tag in case.tags.all():
                tag_name = tag.name
                if tag_name not in tag_counts:
                    tag_counts[tag_name] = 0
                tag_counts[tag_name] += 1

        # Sort by frequency
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

        trends = {}
        for tag_name, count in sorted_tags[:20]:  # Top 20 tags
            trends[tag_name] = {
                'frequency': count,
                'trend_status': 'hot' if count > 10 else 'warm' if count > 5 else 'cool'
            }

        return trends

    def _analyze_temporal_trends(self, cases, time_period: int) -> Dict[str, Any]:
        """Analyze trends over time"""
        # Group by month
        monthly_counts = []

        for months_ago in range(min(12, time_period // 30)):
            month_start = timezone.now() - timedelta(days=30 * (months_ago + 1))
            month_end = timezone.now() - timedelta(days=30 * months_ago)

            count = cases.filter(
                judgment_date__gte=month_start,
                judgment_date__lt=month_end
            ).count()

            monthly_counts.append({
                'month': month_end.strftime('%Y-%m'),
                'case_count': count
            })

        return {
            'monthly_counts': list(reversed(monthly_counts)),
            'average_cases_per_month': sum(item['case_count'] for item in monthly_counts) / len(monthly_counts) if monthly_counts else 0
        }

    def _analyze_outcome_trends(self, cases) -> Dict[str, Any]:
        """Analyze outcome trends"""
        outcomes = {}

        for case in cases:
            outcome = self._determine_case_outcome(case)
            if outcome:
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

        total_cases = sum(outcomes.values()) if outcomes else 1

        trend_data = {}
        for outcome, count in outcomes.items():
            trend_data[outcome] = {
                'count': count,
                'percentage': round((count / total_cases) * 100, 2)
            }

        return trend_data

    def _identify_emerging_topics(self, cases) -> List[Dict[str, Any]]:
        """Identify emerging legal topics"""
        # This would use more sophisticated analysis in practice
        # For now, return recent tags that are gaining traction

        recent_cutoff = timezone.now() - timedelta(days=90)
        recent_cases = cases.filter(judgment_date__gte=recent_cutoff)

        recent_tags = {}
        for case in recent_cases:
            for tag in case.tags.all():
                recent_tags[tag.name] = recent_tags.get(tag.name, 0) + 1

        # Sort by frequency and return top emerging topics
        sorted_topics = sorted(recent_tags.items(), key=lambda x: x[1], reverse=True)

        emerging_topics = []
        for topic, count in sorted_topics[:10]:
            emerging_topics.append({
                'topic': topic,
                'recent_cases': count,
                'growth_rate': 'increasing',  # Would compare with historical data
                'significance': 'high' if count > 5 else 'medium'
            })

        return emerging_topics

    def _determine_case_outcome(self, case: Case) -> str:
        """Determine case outcome for trend analysis"""
        if case.ai_summary and isinstance(case.ai_summary, dict):
            decision = case.ai_summary.get('decision', '').lower()
            if 'allowed' in decision or 'granted' in decision:
                return 'petitioner_favorable'
            elif 'dismissed' in decision or 'rejected' in decision:
                return 'respondent_favorable'

        return 'unknown'

    def _empty_trend_analysis(self) -> Dict[str, Any]:
        """Return empty trend analysis structure"""
        return {
            'case_type_trends': {},
            'court_trends': {},
            'tag_trends': {},
            'temporal_trends': {'monthly_counts': [], 'average_cases_per_month': 0},
            'outcome_trends': {},
            'emerging_topics': [],
            'analysis_metadata': {
                'time_period_days': 0,
                'total_cases': 0,
                'courts_analyzed': 0,
                'analysis_date': datetime.now().isoformat(),
                'status': 'no_data'
            }
        }


class RelevanceScorer:
    """Scores document relevance for legal research queries"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False

    def fit(self, documents: List[str]):
        """Fit the vectorizer on document corpus"""
        try:
            self.vectorizer.fit(documents)
            self.is_fitted = True
            logger.info(f"Relevance scorer fitted on {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to fit relevance scorer: {str(e)}")

    def score_documents(self, query: str, documents: List[str], metadata: List[Dict] = None) -> List[Dict[str, Any]]:
        """Score documents against a query"""
        if not self.is_fitted:
            self.fit(documents)

        try:
            # Vectorize query and documents
            query_vec = self.vectorizer.transform([query])
            doc_vecs = self.vectorizer.transform(documents)

            # Calculate cosine similarity
            similarities = (query_vec * doc_vecs.T).toarray()[0]

            # Combine with metadata if available
            results = []
            for i, (doc, score) in enumerate(zip(documents, similarities)):
                result = {
                    'document_index': i,
                    'relevance_score': float(score),
                    'document_preview': doc[:200] + '...' if len(doc) > 200 else doc
                }

                # Add metadata if provided
                if metadata and i < len(metadata):
                    result.update(metadata[i])

                results.append(result)

            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)

            return results

        except Exception as e:
            logger.error(f"Document scoring failed: {str(e)}")
            return []

    def calculate_case_relevance(self, query: str, case: Case) -> float:
        """Calculate relevance score for a single case"""
        try:
            # Combine relevant text fields
            searchable_text = f"{case.title} {case.headnotes or ''} {case.case_text[:1000]}"

            # Extract keywords from tags
            tag_keywords = ' '.join([tag.name for tag in case.tags.all()])
            searchable_text += f" {tag_keywords}"

            # Score the document
            results = self.score_documents(query, [searchable_text])

            return results[0]['relevance_score'] if results else 0.0

        except Exception as e:
            logger.error(f"Case relevance calculation failed: {str(e)}")
            return 0.0


# Global instances
case_outcome_predictor = CaseOutcomePredictor()
legal_trend_analyzer = LegalTrendAnalyzer()
relevance_scorer = RelevanceScorer()


def train_all_models() -> Dict[str, Any]:
    """Train all ML models"""
    results = {
        'case_outcome_predictor': case_outcome_predictor.train_model(),
        'timestamp': datetime.now().isoformat()
    }

    return results


def predict_case_outcome(case: Case) -> Dict[str, Any]:
    """Predict outcome for a case"""
    features = case_outcome_predictor._extract_case_features(case)
    return case_outcome_predictor.predict(features)


def analyze_legal_trends(time_period: int = 365, court_id: Optional[int] = None) -> Dict[str, Any]:
    """Analyze legal trends"""
    return legal_trend_analyzer.analyze_trends(time_period, court_id)


def score_search_results(query: str, cases: List[Case]) -> List[Dict[str, Any]]:
    """Score search results by relevance"""
    documents = []
    metadata = []

    for case in cases:
        # Create searchable text
        searchable_text = f"{case.title} {case.headnotes or ''} {case.case_text[:1000]}"
        tag_keywords = ' '.join([tag.name for tag in case.tags.all()])
        searchable_text += f" {tag_keywords}"

        documents.append(searchable_text)

        # Prepare metadata
        metadata.append({
            'case_id': str(case.id),
            'case_title': case.title,
            'court': case.court.name,
            'judgment_date': case.judgment_date.isoformat(),
            'tags': [tag.name for tag in case.tags.all()]
        })

    # Score documents
    scored_results = relevance_scorer.score_documents(query, documents, metadata)

    return scored_results