"""
Enhanced Search Engine for CourtVision Pro
Implements semantic search with AI capabilities and hybrid search functionality
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import re
import hashlib

from django.db.models import Q, Count, Avg
from django.core.cache import cache
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
from django.db import transaction

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from .models import Case, HighCourt, Tag, SearchHistory, Customization
from .ai_integration import ai_processor
from .ml_models import relevance_scorer
from django.conf import settings

logger = logging.getLogger(__name__)


class SearchEngineError(Exception):
    """Custom exception for search engine errors"""
    pass


class SemanticSearchEngine:
    """Vector-based semantic search using embeddings"""

    def __init__(self):
        self.model = None
        self.index = None
        self.case_mapping = {}
        self.dimension = 384  # Default for sentence-transformers
        self.is_initialized = False
        self.elasticsearch_client = None

    def initialize(self):
        """Initialize the search engine"""
        try:
            # Load sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = self.model.get_sentence_embedding_dimension()

            # Initialize Elasticsearch if available
            self._initialize_elasticsearch()

            # Build or load FAISS index
            self._build_index()

            self.is_initialized = True
            logger.info("Semantic search engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize semantic search engine: {str(e)}")
            self.is_initialized = False

    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch client"""
        try:
            es_host = getattr(settings, 'ELASTICSEARCH_HOST', 'localhost:9200')
            self.elasticsearch_client = Elasticsearch([es_host])

            # Test connection
            if self.elasticsearch_client.ping():
                logger.info("Elasticsearch connection established")
            else:
                logger.warning("Elasticsearch connection failed")
                self.elasticsearch_client = None

        except Exception as e:
            logger.warning(f"Elasticsearch not available: {str(e)}")
            self.elasticsearch_client = None

    def _build_index(self):
        """Build FAISS index for similarity search"""
        try:
            # Get all cases for indexing
            cases = Case.objects.filter(is_published=True).select_related('court').prefetch_related('tags')

            if not cases.exists():
                logger.warning("No cases found for indexing")
                return

            # Prepare documents for embedding
            documents = []
            case_ids = []

            for case in cases:
                # Create searchable text
                searchable_text = self._prepare_searchable_text(case)
                documents.append(searchable_text)
                case_ids.append(str(case.id))

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.model.encode(documents, show_progress_bar=True)

            # Build FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
            self.index.add(embeddings.astype('float32'))

            # Store mapping
            self.case_mapping = {i: case_id for i, case_id in enumerate(case_ids)}

            logger.info(f"FAISS index built with {len(embeddings)} documents")

            # Also index in Elasticsearch if available
            if self.elasticsearch_client:
                self._index_to_elasticsearch(cases, documents, embeddings)

        except Exception as e:
            logger.error(f"Failed to build search index: {str(e)}")
            raise SearchEngineError(f"Index building failed: {str(e)}")

    def _prepare_searchable_text(self, case: Case) -> str:
        """Prepare searchable text from a case"""
        text_parts = [
            case.title,
            case.citation,
            case.headnotes or '',
            case.case_text[:2000],  # Limit to first 2000 chars for embedding
        ]

        # Add tag keywords
        tag_keywords = ' '.join([tag.name for tag in case.tags.all()])
        text_parts.append(tag_keywords)

        # Add court information
        text_parts.append(f"Court: {case.court.name}")
        text_parts.append(f"Jurisdiction: {case.court.jurisdiction}")

        return ' '.join(text_parts)

    def _index_to_elasticsearch(self, cases, documents, embeddings):
        """Index documents to Elasticsearch"""
        try:
            # Create index if not exists
            index_name = 'legal_cases'
            if not self.elasticsearch_client.indices.exists(index=index_name):
                self.elasticsearch_client.indices.create(index=index_name)

            # Prepare documents for Elasticsearch
            bulk_data = []
            for i, (case, doc, embedding) in enumerate(zip(cases, documents, embeddings)):
                doc_data = {
                    'title': case.title,
                    'citation': case.citation,
                    'headnotes': case.headnotes,
                    'case_text': case.case_text,
                    'court': case.court.name,
                    'jurisdiction': case.court.jurisdiction,
                    'judgment_date': case.judgment_date.isoformat(),
                    'case_type': case.case_type,
                    'tags': [tag.name for tag in case.tags.all()],
                    'searchable_text': doc,
                    'embedding': embedding.tolist()
                }

                bulk_data.append({
                    'index': {'_index': index_name, '_id': str(case.id)}
                })
                bulk_data.append(doc_data)

            # Bulk index
            if bulk_data:
                self.elasticsearch_client.bulk(body=bulk_data)
                logger.info(f"Indexed {len(cases)} documents to Elasticsearch")

        except Exception as e:
            logger.error(f"Elasticsearch indexing failed: {str(e)}")

    async def semantic_search(self, query: str, limit: int = 20, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        if not self.is_initialized:
            self.initialize()

        if not self.is_initialized:
            raise SearchEngineError("Search engine not initialized")

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            k = min(limit * 2, self.index.ntotal)  # Get more results for re-ranking
            distances, indices = self.index.search(query_embedding.astype('float32'), k)

            # Get results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                case_id = self.case_mapping.get(idx)
                if case_id:
                    try:
                        case = Case.objects.get(id=case_id)
                        result = self._prepare_search_result(case, distance, i, query)
                        results.append(result)
                    except Case.DoesNotExist:
                        continue

            # Apply filters
            if filters:
                results = self._apply_filters(results, filters)

            # Sort by relevance score and limit
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            raise SearchEngineError(f"Semantic search failed: {str(e)}")

    def _prepare_search_result(self, case: Case, distance: float, rank: int, query: str) -> Dict[str, Any]:
        """Prepare search result with metadata"""
        # Calculate relevance score from cosine similarity
        relevance_score = float(distance)

        # Extract relevant text snippet
        snippet = self._extract_snippet(case, query)

        return {
            'case_id': str(case.id),
            'title': case.title,
            'citation': case.citation,
            'court': case.court.name,
            'judgment_date': case.judgment_date.isoformat(),
            'case_type': case.case_type,
            'tags': [tag.name for tag in case.tags.all()],
            'snippet': snippet,
            'relevance_score': relevance_score,
            'rank': rank + 1,
            'search_method': 'semantic',
            'highlights': self._extract_highlights(case, query),
            'ai_summary': case.ai_summary.get('summary', '') if case.ai_summary else '',
            'key_principles': case.extracted_principles[:3] if case.extracted_principles else [],
            'view_count': case.view_count,
            'is_bookmarked': False  # Would be set based on user context
        }

    def _extract_snippet(self, case: Case, query: str, max_length: int = 300) -> str:
        """Extract relevant text snippet"""
        # Simple implementation - find query terms in text
        query_terms = query.lower().split()
        text = case.case_text.lower()

        best_snippet = ''
        best_score = 0

        # Search for query terms in the text
        for i in range(0, len(text), 100):  # Slide window
            snippet_start = max(0, i - 50)
            snippet_end = min(len(text), i + max_length + 50)
            snippet = text[snippet_start:snippet_end]

            # Score snippet based on query term matches
            score = sum(1 for term in query_terms if term in snippet)
            if score > best_score:
                best_score = score
                best_snippet = snippet

        if not best_snippet:
            best_snippet = case.headnotes[:max_length] if case.headnotes else case.case_text[:max_length]

        return best_snippet[:max_length] + '...' if len(best_snippet) > max_length else best_snippet

    def _extract_highlights(self, case: Case, query: str) -> List[str]:
        """Extract highlighted text passages"""
        highlights = []
        query_terms = query.lower().split()

        # Search in different text fields
        search_fields = [case.title, case.headnotes or '', case.case_text]

        for field in search_fields:
            field_lower = field.lower()
            for term in query_terms:
                if term in field_lower:
                    # Find context around the term
                    start = field_lower.find(term)
                    context_start = max(0, start - 50)
                    context_end = min(len(field), start + len(term) + 50)
                    context = field[context_start:context_end]

                    if context not in highlights:
                        highlights.append(context)
                        if len(highlights) >= 3:  # Limit highlights
                            break

            if len(highlights) >= 3:
                break

        return highlights

    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply search filters to results"""
        filtered_results = []

        for result in results:
            include = True

            # Court filter
            if 'court' in filters:
                if result['court'] not in filters['court']:
                    include = False

            # Case type filter
            if 'case_type' in filters:
                if result['case_type'] not in filters['case_type']:
                    include = False

            # Date range filter
            if 'date_start' in filters or 'date_end' in filters:
                result_date = datetime.fromisoformat(result['judgment_date']).date()

                if 'date_start' in filters:
                    if result_date < filters['date_start']:
                        include = False

                if 'date_end' in filters:
                    if result_date > filters['date_end']:
                        include = False

            # Tags filter
            if 'tags' in filters and filters['tags']:
                result_tags = set(result['tags'])
                filter_tags = set(filters['tags'])
                if not result_tags.intersection(filter_tags):
                    include = False

            if include:
                filtered_results.append(result)

        return filtered_results


class HybridSearchEngine:
    """Hybrid search combining keyword and semantic search"""

    def __init__(self):
        self.semantic_engine = SemanticSearchEngine()
        self.keyword_weight = 0.3
        self.semantic_weight = 0.7

    async def search(self, query: str, limit: int = 20, filters: Optional[Dict] = None,
                    search_type: str = 'hybrid') -> List[Dict[str, Any]]:
        """Perform search using specified method"""
        try:
            # Log search
            self._log_search(query, filters, limit)

            if search_type == 'keyword':
                return await self._keyword_search(query, limit, filters)
            elif search_type == 'semantic':
                return await self.semantic_engine.semantic_search(query, limit, filters)
            else:  # hybrid
                return await self._hybrid_search(query, limit, filters)

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    async def _keyword_search(self, query: str, limit: int, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Traditional keyword search"""
        try:
            # Build database query
            db_query = Q()

            # Search in multiple fields
            search_terms = query.split()
            for term in search_terms:
                db_query |= (
                    Q(title__icontains=term) |
                    Q(headnotes__icontains=term) |
                    Q(case_text__icontains=term) |
                    Q(citation__icontains=term)
                )

            # Add tag search
            db_query |= Q(tags__name__icontains=query)

            # Base queryset
            cases = Case.objects.filter(db_query, is_published=True).select_related('court').prefetch_related('tags').distinct()

            # Apply filters
            if filters:
                cases = self._apply_database_filters(cases, filters)

            # Order by relevance (simple implementation)
            cases = cases.order_by('-view_count', '-judgment_date')[:limit * 2]

            # Convert to results format
            results = []
            for i, case in enumerate(cases):
                result = self._prepare_keyword_search_result(case, i, query)
                results.append(result)

            return results[:limit]

        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []

    async def _hybrid_search(self, query: str, limit: int, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Hybrid search combining keyword and semantic"""
        try:
            # Get results from both methods
            keyword_results = await self._keyword_search(query, limit * 2, filters)
            semantic_results = await self.semantic_engine.semantic_search(query, limit * 2, filters)

            # Combine and re-rank results
            combined_results = self._combine_search_results(keyword_results, semantic_results)

            return combined_results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            # Fallback to keyword search
            return await self._keyword_search(query, limit, filters)

    def _combine_search_results(self, keyword_results: List[Dict], semantic_results: List[Dict]) -> List[Dict]:
        """Combine and re-rank results from multiple search methods"""
        # Create a dictionary to store combined results
        combined = {}

        # Process keyword results
        for result in keyword_results:
            case_id = result['case_id']
            combined[case_id] = result.copy()
            combined[case_id]['keyword_score'] = 1.0 - (result['rank'] * 0.1)  # Simple decay
            combined[case_id]['semantic_score'] = 0.0

        # Process semantic results
        for result in semantic_results:
            case_id = result['case_id']
            if case_id in combined:
                combined[case_id]['semantic_score'] = result['relevance_score']
                combined[case_id]['highlights'] = result.get('highlights', combined[case_id].get('highlights', []))
            else:
                combined[case_id] = result.copy()
                combined[case_id]['keyword_score'] = 0.0
                combined[case_id]['semantic_score'] = result['relevance_score']

        # Calculate final scores
        for case_id, result in combined.items():
            final_score = (
                self.keyword_weight * result['keyword_score'] +
                self.semantic_weight * result['semantic_score']
            )
            result['final_score'] = final_score
            result['search_method'] = 'hybrid'

        # Sort by final score
        sorted_results = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)

        # Update ranks
        for i, result in enumerate(sorted_results):
            result['rank'] = i + 1

        return sorted_results

    def _prepare_keyword_search_result(self, case: Case, rank: int, query: str) -> Dict[str, Any]:
        """Prepare keyword search result"""
        snippet = self.semantic_engine._extract_snippet(case, query)
        highlights = self.semantic_engine._extract_highlights(case, query)

        return {
            'case_id': str(case.id),
            'title': case.title,
            'citation': case.citation,
            'court': case.court.name,
            'judgment_date': case.judgment_date.isoformat(),
            'case_type': case.case_type,
            'tags': [tag.name for tag in case.tags.all()],
            'snippet': snippet,
            'relevance_score': 1.0 - (rank * 0.1),  # Simple relevance decay
            'rank': rank + 1,
            'search_method': 'keyword',
            'highlights': highlights,
            'ai_summary': case.ai_summary.get('summary', '') if case.ai_summary else '',
            'key_principles': case.extracted_principles[:3] if case.extracted_principles else [],
            'view_count': case.view_count,
            'is_bookmarked': False
        }

    def _apply_database_filters(self, queryset, filters: Dict):
        """Apply filters to Django queryset"""
        if 'court' in filters:
            queryset = queryset.filter(court__name__in=filters['court'])

        if 'case_type' in filters:
            queryset = queryset.filter(case_type__in=filters['case_type'])

        if 'date_start' in filters:
            queryset = queryset.filter(judgment_date__gte=filters['date_start'])

        if 'date_end' in filters:
            queryset = queryset.filter(judgment_date__lte=filters['date_end'])

        if 'tags' in filters and filters['tags']:
            queryset = queryset.filter(tags__name__in=filters['tags'])

        return queryset

    def _log_search(self, query: str, filters: Optional[Dict], limit: int):
        """Log search for analytics"""
        try:
            # This would be implemented with actual user context
            # For now, just log to the system
            logger.info(f"Search query: {query}, filters: {filters}, limit: {limit}")
        except Exception as e:
            logger.warning(f"Failed to log search: {str(e)}")


class QueryExpander:
    """Expands and enhances search queries"""

    def __init__(self):
        self.legal_synonyms = {
            'contract': ['agreement', 'treaty', 'pact', 'compact'],
            'breach': ['violation', 'infraction', 'contravention'],
            'damages': ['compensation', 'reparation', 'indemnity'],
            'injunction': ['order', 'decree', 'directive', 'mandate'],
            'liability': ['responsibility', 'accountability', 'obligation'],
            'negligence': ['carelessness', 'neglect', 'oversight'],
            'fraud': ['deception', 'misrepresentation', 'trickery'],
            'appeal': ['review', 'challenge', 'contest'],
            'dismiss': ['reject', 'deny', 'refute'],
            'grant': ['allow', 'permit', 'approve']
        }

    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded_queries = [query]
        query_lower = query.lower()

        # Add synonym-based queries
        for term, synonyms in self.legal_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    if expanded_query not in [q.lower() for q in expanded_queries]:
                        expanded_queries.append(expanded_query)

        return expanded_queries

    def suggest_corrections(self, query: str) -> List[str]:
        """Suggest query corrections (simple implementation)"""
        corrections = []

        # Common legal term corrections
        corrections_map = {
            'contempt': 'contempt of court',
            'injuction': 'injunction',
            'liablity': 'liability',
            'necligence': 'negligence',
            'commecial': 'commercial',
        }

        query_lower = query.lower()
        for wrong_term, correct_term in corrections_map.items():
            if wrong_term in query_lower:
                corrected_query = query_lower.replace(wrong_term, correct_term)
                corrections.append(corrected_query)

        return corrections


# Global search engine instance
search_engine = HybridSearchEngine()
query_expander = QueryExpander()


async def perform_search(query: str, user=None, limit: int = 20, filters: Optional[Dict] = None,
                        search_type: str = 'hybrid') -> Dict[str, Any]:
    """Main search function"""
    try:
        # Expand query if needed
        if search_type in ['hybrid', 'keyword']:
            expanded_queries = query_expander.expand_query(query)
            if len(expanded_queries) > 1:
                # Use original query for now, but could try multiple queries
                pass

        # Perform search
        results = await search_engine.search(query, limit, filters, search_type)

        # Apply user customization if available
        if user:
            results = apply_user_customization(results, user)

        # Get query suggestions
        suggestions = query_expander.suggest_corrections(query)

        return {
            'results': results,
            'total_results': len(results),
            'query': query,
            'search_type': search_type,
            'suggestions': suggestions,
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'filters_applied': filters or {},
                'processing_time': 0.0  # Would be calculated
            }
        }

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return {
            'results': [],
            'total_results': 0,
            'query': query,
            'error': str(e),
            'search_type': search_type
        }


def apply_user_customization(results: List[Dict], user) -> List[Dict]:
    """Apply user customization to search results"""
    try:
        # Get user's customization preferences
        customization = Customization.objects.filter(user=user).first()
        if not customization:
            return results

        # Apply jurisdiction emphasis
        if customization.jurisdiction_emphasis:
            emphasized_courts = customization.jurisdiction_emphasis.get('emphasized_courts', [])
            for result in results:
                if result['court'] in emphasized_courts:
                    result['relevance_score'] *= 1.2  # Boost emphasized courts

        # Apply time period focus
        if customization.time_period_focus != 'historical':
            current_year = datetime.now().year
            if customization.time_period_focus == 'recent':
                cutoff_year = current_year - 5
            elif customization.time_period_focus == 'medium':
                cutoff_year = current_year - 10
            else:  # custom
                cutoff_year = current_year - 15  # Default

            for result in results:
                result_year = datetime.fromisoformat(result['judgment_date']).year
                if result_year < cutoff_year:
                    result['relevance_score'] *= 0.8  # Penalize older cases

        # Re-sort by modified relevance scores
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1

        return results

    except Exception as e:
        logger.error(f"Failed to apply user customization: {str(e)}")
        return results


def rebuild_search_index():
    """Rebuild the search index"""
    try:
        search_engine.semantic_engine.initialize()
        return {'success': True, 'message': 'Search index rebuilt successfully'}
    except Exception as e:
        logger.error(f"Failed to rebuild search index: {str(e)}")
        return {'success': False, 'message': str(e)}