"""
AI Integration Module for CourtVision Pro
Handles AI service integrations, text processing, and legal document analysis
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
import uuid

import openai
from openai import AsyncOpenAI
import spacy
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import redis

from django.conf import settings
from django.core.cache import cache
from .models import Case, HighCourt, Customization

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Custom exception for AI service errors"""
    pass


class AIServiceClient:
    """Base class for AI service clients"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.is_available = True
        self.last_check = datetime.now()
        self.error_count = 0
        self.max_errors = 5

    def check_availability(self) -> bool:
        """Check if the AI service is available"""
        if self.error_count >= self.max_errors:
            self.is_available = False
            if datetime.now() - self.last_check > timedelta(minutes=5):
                self.is_available = True
                self.error_count = 0
        return self.is_available

    def handle_error(self, error: Exception):
        """Handle service errors and implement circuit breaker"""
        self.error_count += 1
        self.last_check = datetime.now()
        logger.error(f"AI Service {self.service_name} error: {str(error)}")

        if self.error_count >= self.max_errors:
            self.is_available = False
            logger.warning(f"AI Service {self.service_name} marked as unavailable")


class OpenAIClient(AIServiceClient):
    """OpenAI API client for advanced AI tasks"""

    def __init__(self):
        super().__init__("OpenAI")
        self.client = AsyncOpenAI(api_key=getattr(settings, 'OPENAI_API_KEY', None))
        self.model = getattr(settings, 'OPENAI_MODEL', 'gpt-4-turbo-preview')

    async def extract_legal_principles(self, document_text: str) -> List[Dict[str, Any]]:
        """Extract key legal principles from document text"""
        if not self.check_availability():
            raise AIServiceError("OpenAI service is currently unavailable")

        try:
            prompt = f"""
            Extract the key legal principles from this legal document.
            Return them as a JSON array where each principle has:
            - principle: The legal principle statement
            - context: Brief context/explanation
            - confidence: Confidence score (0-1)

            Document text:
            {document_text[:4000]}
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal expert AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            result = response.choices[0].message.content
            principles = json.loads(result)

            # Cache the result
            cache_key = f"principles_{hashlib.md5(document_text.encode()).hexdigest()}"
            cache.set(cache_key, principles, timeout=86400)  # 24 hours

            return principles

        except Exception as e:
            self.handle_error(e)
            raise AIServiceError(f"Failed to extract legal principles: {str(e)}")

    async def identify_precedents(self, case_text: str, case_database: List[Dict]) -> List[Dict[str, Any]]:
        """Identify relevant precedents for a given case"""
        if not self.check_availability():
            raise AIServiceError("OpenAI service is currently unavailable")

        try:
            # Create a summary of case database for context
            case_summary = "\n".join([
                f"Case {i+1}: {case.get('title', '')} - {case.get('citation', '')} - {case.get('summary', '')[:200]}"
                for i, case in enumerate(case_database[:10])
            ])

            prompt = f"""
            Given the current case and a database of precedents, identify the most relevant precedents.
            Return a JSON array with:
            - case_id: ID of the precedent case
            - relevance_score: Score 0-1 indicating relevance
            - reasoning: Why this precedent is relevant
            - legal_principles: Common legal principles

            Current Case:
            {case_text[:2000]}

            Precedent Database:
            {case_summary}
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal research expert AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            result = response.choices[0].message.content
            precedents = json.loads(result)

            return precedents

        except Exception as e:
            self.handle_error(e)
            raise AIServiceError(f"Failed to identify precedents: {str(e)}")

    async def generate_case_summary(self, full_text: str, customization: Optional[Customization] = None) -> Dict[str, Any]:
        """Generate customized AI summary of a legal case"""
        if not self.check_availability():
            raise AIServiceError("OpenAI service is currently unavailable")

        try:
            # Build customization context
            custom_context = ""
            if customization:
                focus_areas = customization.analysis_focus_areas or []
                if focus_areas:
                    custom_context = f"\nFocus on these areas: {', '.join(focus_areas)}"

                weight = customization.precedent_statute_weight
                if weight > 0.6:
                    custom_context += "\nEmphasize statutory interpretation over precedent."
                elif weight < 0.4:
                    custom_context += "\nEmphasize case law precedent over statutory interpretation."

            prompt = f"""
            Generate a comprehensive summary of this legal case. The summary should include:
            - summary: Brief overview of the case
            - key_points: Main legal issues and decisions
            - decision: Final judgment and its implications
            - implications: Broader legal implications
            - statutes_cited: Key statutes mentioned
            - precedents_cited: Important precedents referenced

            {custom_context}

            Case text:
            {full_text[:4000]}
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal expert providing case summaries for judicial officers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            result = response.choices[0].message.content
            summary_data = json.loads(result)

            return summary_data

        except Exception as e:
            self.handle_error(e)
            raise AIServiceError(f"Failed to generate case summary: {str(e)}")


class LocalModelClient(AIServiceClient):
    """Local model client for offline processing"""

    def __init__(self):
        super().__init__("LocalModels")
        self.tokenizer = None
        self.model = None
        self.nlp = None
        self._load_models()

    def _load_models(self):
        """Load local NLP models"""
        try:
            # Load spaCy model for basic NLP
            self.nlp = spacy.load("en_core_web_sm")

            # Load transformer model for embeddings (lighter model)
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        except Exception as e:
            logger.warning(f"Failed to load local models: {str(e)}")
            self.is_available = False

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal entities using spaCy"""
        if not self.nlp or not self.check_availability():
            return []

        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })

            return entities

        except Exception as e:
            self.handle_error(e)
            return []

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate text embeddings using local transformer model"""
        if not self.model or not self.check_availability():
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000)
            return vectorizer.fit_transform(texts).toarray()

        try:
            # Tokenize texts
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = model_output.last_hidden_state.mean(dim=1)

            return embeddings.numpy()

        except Exception as e:
            self.handle_error(e)
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000)
            return vectorizer.fit_transform(texts).toarray()


class LegalTextProcessor:
    """Main processor for legal document analysis"""

    def __init__(self):
        self.openai_client = OpenAIClient()
        self.local_client = LocalModelClient()
        self.cache_timeout = 86400  # 24 hours

    async def process_legal_document(self, case: Case) -> Dict[str, Any]:
        """Process a legal document and extract AI insights"""
        try:
            document_text = case.case_text

            # Check cache first
            cache_key = f"ai_processing_{case.id}"
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result

            # Initialize results
            results = {
                'principles': [],
                'precedents': [],
                'summary': {},
                'entities': [],
                'embeddings': None,
                'processing_metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'services_used': [],
                    'model_versions': {}
                }
            }

            # Try OpenAI first for advanced processing
            if self.openai_client.check_availability():
                try:
                    # Extract legal principles
                    principles = await self.openai_client.extract_legal_principles(document_text)
                    results['principles'] = principles
                    results['processing_metadata']['services_used'].append('openai_principles')

                    # Generate summary
                    customization = self._get_user_customization(case)
                    summary = await self.openai_client.generate_case_summary(document_text, customization)
                    results['summary'] = summary
                    results['processing_metadata']['services_used'].append('openai_summary')

                    # Find precedents (sample implementation)
                    similar_cases = self._get_similar_cases(case, limit=10)
                    if similar_cases:
                        precedents = await self.openai_client.identify_precedents(document_text, similar_cases)
                        results['precedents'] = precedents
                        results['processing_metadata']['services_used'].append('openai_precedents')

                except Exception as e:
                    logger.warning(f"OpenAI processing failed: {str(e)}")

            # Use local models for basic processing
            if self.local_client.check_availability():
                try:
                    # Extract entities
                    entities = self.local_client.extract_entities(document_text)
                    results['entities'] = entities
                    results['processing_metadata']['services_used'].append('local_entities')

                    # Generate embeddings
                    embeddings = self.local_client.generate_embeddings([document_text])
                    results['embeddings'] = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
                    results['processing_metadata']['services_used'].append('local_embeddings')

                except Exception as e:
                    logger.warning(f"Local model processing failed: {str(e)}")

            # Cache results
            cache.set(cache_key, results, timeout=self.cache_timeout)

            return results

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise AIServiceError(f"Failed to process legal document: {str(e)}")

    def _get_user_customization(self, case: Case) -> Optional[Customization]:
        """Get user customization for the case"""
        try:
            # This would be implemented based on current user context
            # For now, return None as placeholder
            return None
        except:
            return None

    def _get_similar_cases(self, case: Case, limit: int = 10) -> List[Dict]:
        """Get similar cases for precedent analysis"""
        try:
            # Find cases with same tags or court
            similar_cases = Case.objects.filter(
                court=case.court
            ).exclude(id=case.id)[:limit]

            return [
                {
                    'id': str(c.id),
                    'title': c.title,
                    'citation': c.citation,
                    'summary': c.headnotes or '',
                    'judgment_date': c.judgment_date.isoformat()
                }
                for c in similar_cases
            ]
        except Exception as e:
            logger.warning(f"Failed to get similar cases: {str(e)}")
            return []


class PredictiveAnalytics:
    """Predictive analytics for case outcomes"""

    def __init__(self):
        self.openai_client = OpenAIClient()

    async def predict_case_outcome(self, case_features: Dict[str, Any], historical_data: List[Dict]) -> Dict[str, Any]:
        """Predict case outcome based on features and historical data"""
        if not self.openai_client.check_availability():
            return self._fallback_prediction(case_features)

        try:
            # Prepare historical context
            history_summary = self._prepare_historical_summary(historical_data)

            prompt = f"""
            Based on historical case data and current case features, predict the likely outcome.
            Provide:
            - predicted_outcome: Likely judgment
            - confidence: Confidence score (0-1)
            - key_factors: Main factors influencing prediction
            - similar_cases: Historical cases with similar outcomes
            - risk_assessment: Risk level and factors

            Current Case Features:
            {json.dumps(case_features, indent=2)}

            Historical Data Summary:
            {history_summary}
            """

            response = await self.openai_client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": "You are a legal analytics expert providing case outcome predictions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            result = response.choices[0].message.content
            prediction = json.loads(result)

            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._fallback_prediction(case_features)

    def _prepare_historical_summary(self, historical_data: List[Dict]) -> str:
        """Prepare summary of historical cases"""
        if not historical_data:
            return "No historical data available."

        summary_lines = []
        for i, case in enumerate(historical_data[:10]):
            summary_lines.append(
                f"Case {i+1}: {case.get('title', '')} - Outcome: {case.get('outcome', '')} - "
                f"Duration: {case.get('duration', '')} - Court: {case.get('court', '')}"
            )

        return "\n".join(summary_lines)

    def _fallback_prediction(self, case_features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction when AI services are unavailable"""
        return {
            'predicted_outcome': 'Unable to predict due to service limitations',
            'confidence': 0.0,
            'key_factors': ['AI service unavailable'],
            'similar_cases': [],
            'risk_assessment': 'Cannot assess risk at this time',
            'fallback_used': True
        }


# Global instances
ai_processor = LegalTextProcessor()
predictive_analytics = PredictiveAnalytics()


async def process_case_ai(case: Case) -> Dict[str, Any]:
    """Main function to process a case with AI"""
    return await ai_processor.process_legal_document(case)


async def predict_case_outcome(case: Case, historical_data: List[Dict] = None) -> Dict[str, Any]:
    """Main function to predict case outcome"""
    case_features = {
        'title': case.title,
        'court': case.court.name,
        'case_type': case.case_type,
        'judgment_date': case.judgment_date.isoformat(),
        'tags': [tag.name for tag in case.tags.all()],
        'citation': case.citation
    }

    if historical_data is None:
        historical_data = []

    return await predictive_analytics.predict_case_outcome(case_features, historical_data)