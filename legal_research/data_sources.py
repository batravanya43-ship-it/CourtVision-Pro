"""
Data Sources Integration for CourtVision Pro
Integration with real legal data sources and automated data import
"""

import json
import logging
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import hashlib
import re
from pathlib import Path

from django.db import transaction
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from .models import Case, HighCourt, Tag
from .ai_integration import ai_processor

logger = logging.getLogger(__name__)


class DataSourcesError(Exception):
    """Custom exception for data sources errors"""
    pass


class LegalDataSource:
    """Base class for legal data sources"""

    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.session = None
        self.rate_limit_delay = 1  # seconds between requests
        self.last_request_time = None
        self.request_count = 0
        self.max_requests_per_hour = 1000

    async def initialize(self):
        """Initialize the data source"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'CourtVision-Pro Legal Research Bot 1.0'}
        )
        logger.info(f"Initialized data source: {self.name}")

    async def close(self):
        """Close the data source session"""
        if self.session:
            await self.session.close()
            logger.info(f"Closed data source: {self.name}")

    async def _rate_limit(self):
        """Implement rate limiting"""
        if self.last_request_time:
            elapsed = datetime.now() - self.last_request_time
            if elapsed.total_seconds() < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed.total_seconds())

        self.last_request_time = datetime.now()
        self.request_count += 1

        # Check hourly rate limit
        if self.request_count >= self.max_requests_per_hour:
            logger.warning(f"Rate limit reached for {self.name}")
            await asyncio.sleep(3600)  # Wait 1 hour
            self.request_count = 0

    async def fetch_data(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch data from the source"""
        await self._rate_limit()

        try:
            url = urljoin(self.base_url, endpoint)
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"HTTP {response.status} from {url}")
                    return None

        except Exception as e:
            logger.error(f"Failed to fetch data from {self.name}: {str(e)}")
            return None

    def validate_data(self, data: Dict) -> bool:
        """Validate fetched data"""
        required_fields = ['title', 'citation', 'judgment_date']
        return all(field in data for field in required_fields)


class SupremeCourtDataSource(LegalDataSource):
    """Supreme Court of India judgments data source"""

    def __init__(self):
        super().__init__(
            "Supreme Court of India",
            "https://main.sci.gov.in"
        )
        self.endpoints = {
            'judgments': '/judgment',
            'case_status': '/case-status',
            'daily_orders': '/daily-order'
        }

    async def fetch_recent_judgments(self, days: int = 30) -> List[Dict]:
        """Fetch recent Supreme Court judgments"""
        try:
            judgments = []

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Fetch judgments (simplified implementation)
            # In reality, this would parse the Supreme Court website
            for i in range(10):  # Limit to 10 for demo
                judgment_data = {
                    'title': f'Supreme Court Judgment {i+1}',
                    'citation': f'2024 SCC OnLine SC {i+1}',
                    'judgment_date': (start_date + timedelta(days=i*3)).date().isoformat(),
                    'decision_date': (start_date + timedelta(days=i*3+1)).date().isoformat(),
                    'court': 'Supreme Court of India',
                    'petitioners': f'Petitioner {i+1}',
                    'respondents': f'Respondent {i+1}',
                    'case_text': f'This is the full text of Supreme Court judgment {i+1}...',
                    'headnotes': f'Key legal principles from judgment {i+1}',
                    'case_type': 'judgment',
                    'source': 'supreme_court',
                    'source_id': f'sc_{i+1}',
                    'source_url': f'{self.base_url}/judgment/{i+1}'
                }

                if self.validate_data(judgment_data):
                    judgments.append(judgment_data)

            logger.info(f"Fetched {len(judgments)} Supreme Court judgments")
            return judgments

        except Exception as e:
            logger.error(f"Failed to fetch Supreme Court judgments: {str(e)}")
            return []


class HighCourtDataSource(LegalDataSource):
    """High Court judgments data source"""

    def __init__(self, court_name: str, base_url: str):
        super().__init__(court_name, base_url)
        self.court_name = court_name

    async def fetch_recent_judgments(self, days: int = 30) -> List[Dict]:
        """Fetch recent High Court judgments"""
        try:
            judgments = []

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Generate sample judgments (simplified implementation)
            for i in range(5):  # Limit to 5 for demo
                judgment_data = {
                    'title': f'{self.court_name} Judgment {i+1}',
                    'citation': f'2024 {self.court_name.split()[0]} HC {i+1}',
                    'judgment_date': (start_date + timedelta(days=i*5)).date().isoformat(),
                    'decision_date': (start_date + timedelta(days=i*5+2)).date().isoformat(),
                    'court': self.court_name,
                    'petitioners': f'Petitioner {i+1}',
                    'respondents': f'Respondent {i+1}',
                    'case_text': f'This is the full text of {self.court_name} judgment {i+1}...',
                    'headnotes': f'Key legal principles from {self.court_name} judgment {i+1}',
                    'case_type': 'judgment',
                    'source': 'high_court',
                    'source_id': f'hc_{self.court_name}_{i+1}',
                    'source_url': f'{self.base_url}/judgment/{i+1}'
                }

                if self.validate_data(judgment_data):
                    judgments.append(judgment_data)

            logger.info(f"Fetched {len(judgments)} {self.court_name} judgments")
            return judgments

        except Exception as e:
            logger.error(f"Failed to fetch {self.court_name} judgments: {str(e)}")
            return []


class LegalDatabaseAPI(LegalDataSource):
    """Legal database API integration (e.g., Manupatra, SCC Online)"""

    def __init__(self, name: str, api_key: str, base_url: str):
        super().__init__(name, base_url)
        self.api_key = api_key
        self.auth_headers = {'Authorization': f'Bearer {api_key}'}

    async def search_cases(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for cases using the legal database API"""
        try:
            params = {
                'q': query,
                'limit': limit,
                'format': 'json'
            }

            # Update session headers with auth
            if self.session:
                self.session.headers.update(self.auth_headers)

            data = await self.fetch_data('/search', params)
            if data and 'results' in data:
                return data['results']

            return []

        except Exception as e:
            logger.error(f"Failed to search cases in {self.name}: {str(e)}")
            return []

    async def get_case_details(self, case_id: str) -> Optional[Dict]:
        """Get detailed case information"""
        try:
            # Update session headers with auth
            if self.session:
                self.session.headers.update(self.auth_headers)

            data = await self.fetch_data(f'/cases/{case_id}')
            return data

        except Exception as e:
            logger.error(f"Failed to get case details from {self.name}: {str(e)}")
            return None


class PDFDocumentProcessor:
    """Process PDF documents from legal sources"""

    def __init__(self):
        self.supported_formats = ['.pdf']
        self.max_file_size = 50 * 1024 * 1024  # 50MB

    def process_pdf(self, pdf_path: str) -> Optional[Dict]:
        """Extract text and metadata from PDF"""
        try:
            if not PDF_AVAILABLE:
                logger.error("pdfplumber not available")
                return None

            # Check file size
            file_size = Path(pdf_path).stat().st_size
            if file_size > self.max_file_size:
                logger.error(f"PDF file too large: {file_size} bytes")
                return None

            with pdfplumber.open(pdf_path) as pdf:
                text_content = []
                metadata = {}

                # Extract text from all pages
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)

                # Extract metadata
                if pdf.metadata:
                    metadata.update(pdf.metadata)

                full_text = '\n'.join(text_content)

                return {
                    'text': full_text,
                    'metadata': metadata,
                    'page_count': len(pdf.pages),
                    'processed_at': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
            return None

    def extract_case_metadata(self, text: str) -> Dict[str, Any]:
        """Extract case metadata from text"""
        try:
            metadata = {}

            # Extract case title (simplified pattern)
            title_patterns = [
                r'(?:IN THE\s+(?:SUPREME COURT|HIGH COURT)[\s\S]*?\n)([\s\S]*?)\n\s*vs\.?\s*\n',
                r'(?:Title\s*:?)([\s\S]*?)(?:\n\s*vs\.?|\n\s*Citation)',
                r'^([A-Z][^.]*\.)\s*(?:vs\.?|versus)'
            ]

            for pattern in title_patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    metadata['title'] = match.group(1).strip()
                    break

            # Extract citation
            citation_patterns = [
                r'(?:Citation\s*:?)([\w\s\-\./]+?)(?:\n|$)',
                r'(\d{4}\s+(?:SCC|AIR|SCR)\s+[\w\s\-\./]+?)(?:\n|$)'
            ]

            for pattern in citation_patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    metadata['citation'] = match.group(1).strip()
                    break

            # Extract judgment date
            date_patterns = [
                r'(?:Date\s*:?|Judgment\s*date\s*:?)(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})',
                r'(?:Dated\s*:?\s*)(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})'
            ]

            for pattern in date_patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    try:
                        date_str = match.group(1).strip()
                        # Try to parse date
                        for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y']:
                            try:
                                parsed_date = datetime.strptime(date_str, fmt)
                                metadata['judgment_date'] = parsed_date.date().isoformat()
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass
                    break

            # Extract court name
            court_patterns = [
                r'(?:IN THE\s+(SUPREME COURT OF INDIA|[\w\s]+HIGH COURT))',
                r'(?:BEFORE\s+THE\s+(?:HON\'BLE\s+)?([\w\s]+COURT))'
            ]

            for pattern in court_patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    metadata['court'] = match.group(1).strip()
                    break

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract case metadata: {str(e)}")
            return {}


class DataImportManager:
    """Manages data import from various sources"""

    def __init__(self):
        self.data_sources = {}
        self.pdf_processor = PDFDocumentProcessor()
        self.import_stats = {
            'total_cases_imported': 0,
            'failed_imports': 0,
            'last_import': None
        }

    def register_data_source(self, source: LegalDataSource):
        """Register a data source"""
        self.data_sources[source.name] = source
        logger.info(f"Registered data source: {source.name}")

    async def import_from_all_sources(self, days: int = 30) -> Dict[str, Any]:
        """Import data from all registered sources"""
        try:
            import_results = {}
            total_imported = 0
            total_failed = 0

            for source_name, source in self.data_sources.items():
                try:
                    logger.info(f"Importing from {source_name}")
                    await source.initialize()

                    if isinstance(source, SupremeCourtDataSource):
                        cases = await source.fetch_recent_judgments(days)
                    elif isinstance(source, HighCourtDataSource):
                        cases = await source.fetch_recent_judgments(days)
                    else:
                        cases = []

                    # Process cases
                    import_result = await self.process_imported_cases(cases, source_name)
                    import_results[source_name] = import_result

                    total_imported += import_result['imported']
                    total_failed += import_result['failed']

                    await source.close()

                except Exception as e:
                    logger.error(f"Failed to import from {source_name}: {str(e)}")
                    import_results[source_name] = {
                        'imported': 0,
                        'failed': 0,
                        'error': str(e)
                    }
                    total_failed += 1

            # Update stats
            self.import_stats.update({
                'total_cases_imported': self.import_stats['total_cases_imported'] + total_imported,
                'failed_imports': self.import_stats['failed_imports'] + total_failed,
                'last_import': datetime.now().isoformat()
            })

            result = {
                'import_summary': {
                    'total_imported': total_imported,
                    'total_failed': total_failed,
                    'sources_processed': len(import_results),
                    'import_date': datetime.now().isoformat()
                },
                'source_results': import_results,
                'overall_stats': self.import_stats
            }

            logger.info(f"Import completed: {total_imported} cases imported, {total_failed} failed")
            return result

        except Exception as e:
            logger.error(f"Data import failed: {str(e)}")
            return {'error': str(e), 'import_summary': {}}

    async def process_imported_cases(self, cases: List[Dict], source_name: str) -> Dict[str, Any]:
        """Process imported cases and save to database"""
        imported = 0
        failed = 0

        try:
            for case_data in cases:
                try:
                    with transaction.atomic():
                        # Check if case already exists
                        existing_case = Case.objects.filter(
                            citation=case_data.get('citation', '')
                        ).first()

                        if existing_case:
                            logger.debug(f"Case already exists: {case_data.get('citation')}")
                            continue

                        # Get or create High Court
                        court_name = case_data.get('court', 'Unknown Court')
                        court, created = HighCourt.objects.get_or_create(
                            name=court_name,
                            defaults={
                                'jurisdiction': 'India',
                                'code': court_name.replace(' ', '_').lower()[:10],
                                'established_date': datetime.now().date(),
                                'is_active': True
                            }
                        )

                        # Create tags
                        tags = []
                        tag_names = self._extract_tags_from_case(case_data)
                        for tag_name in tag_names:
                            tag, created = Tag.objects.get_or_create(
                                name=tag_name,
                                defaults={'description': f'Auto-generated tag: {tag_name}'}
                            )
                            tags.append(tag)

                        # Create case
                        case = Case.objects.create(
                            title=case_data.get('title', ''),
                            citation=case_data.get('citation', ''),
                            court=court,
                            bench='',
                            judgment_date=datetime.fromisoformat(case_data.get('judgment_date', datetime.now().isoformat())).date(),
                            decision_date=datetime.fromisoformat(case_data.get('decision_date', datetime.now().isoformat())).date(),
                            petitioners=case_data.get('petitioners', ''),
                            respondents=case_data.get('respondents', ''),
                            case_text=case_data.get('case_text', ''),
                            headnotes=case_data.get('headnotes', ''),
                            case_type=case_data.get('case_type', 'judgment'),
                            relevance_score=0.0,
                            is_published=True
                        )

                        # Add tags to case
                        if tags:
                            case.tags.add(*tags)

                        # Process with AI (if available)
                        try:
                            ai_summary = await ai_processor.process_legal_document(case)
                            if ai_summary:
                                case.ai_summary = ai_summary.get('summary', {})
                                case.extracted_principles = ai_summary.get('principles', [])
                                case.statutes_cited = ai_summary.get('statutes_cited', [])
                                case.precedents_cited = ai_summary.get('precedents', [])
                                case.save()
                        except Exception as e:
                            logger.warning(f"AI processing failed for case {case.id}: {str(e)}")

                        imported += 1
                        logger.debug(f"Imported case: {case.citation}")

                except Exception as e:
                    logger.error(f"Failed to import case {case_data.get('citation', 'unknown')}: {str(e)}")
                    failed += 1

        except Exception as e:
            logger.error(f"Batch case processing failed: {str(e)}")
            failed += len(cases)

        return {
            'imported': imported,
            'failed': failed,
            'total_processed': len(cases)
        }

    def _extract_tags_from_case(self, case_data: Dict) -> List[str]:
        """Extract tags from case data"""
        tags = []

        # Extract from title and text
        text_content = f"{case_data.get('title', '')} {case_data.get('headnotes', '')} {case_data.get('case_text', '')}"

        # Legal topic keywords
        legal_topics = [
            'contract', 'breach', 'damages', 'injunction', 'specific performance',
            'company law', 'insolvency', 'bankruptcy', 'merger', 'acquisition',
            'intellectual property', 'trademark', 'copyright', 'patent',
            'taxation', 'income tax', 'gst', 'customs duty',
            'labor law', 'employment', 'termination', 'wages',
            'property law', 'land acquisition', 'rent control', 'easement',
            'constitutional law', 'fundamental rights', 'directive principles',
            'criminal law', 'bail', 'anticipatory bail', 'quashing',
            'civil procedure', 'appeal', 'revision', 'review'
        ]

        for topic in legal_topics:
            if topic.lower() in text_content.lower():
                tags.append(topic.title())

        # Extract court-specific tags
        court = case_data.get('court', '')
        if 'supreme court' in court.lower():
            tags.append('Supreme Court')
        elif 'high court' in court.lower():
            tags.append('High Court')

        # Add source tag
        source = case_data.get('source', '')
        if source:
            tags.append(f"Source: {source.title()}")

        return list(set(tags))  # Remove duplicates


# Global data import manager
data_import_manager = DataImportManager()


def initialize_data_sources():
    """Initialize all data sources"""
    try:
        # Register Supreme Court data source
        supreme_court_source = SupremeCourtDataSource()
        data_import_manager.register_data_source(supreme_court_source)

        # Register High Court data sources
        high_courts = [
            ('Delhi High Court', 'https://delhihighcourt.nic.in'),
            ('Bombay High Court', 'https://bombayhighcourt.nic.in'),
            ('Calcutta High Court', 'https://calcuttahighcourt.nic.in'),
            ('Madras High Court', 'https://madhighcourt.nic.in'),
        ]

        for court_name, base_url in high_courts:
            high_court_source = HighCourtDataSource(court_name, base_url)
            data_import_manager.register_data_source(high_court_source)

        # Initialize legal database APIs (would require API keys)
        # Example: manupatra_api = LegalDatabaseAPI('Manupatra', 'api_key', 'https://api.manupatra.com')
        # data_import_manager.register_data_source(manupatra_api)

        logger.info("Data sources initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize data sources: {str(e)}")
        return False


async def perform_data_import(days: int = 30) -> Dict[str, Any]:
    """Perform data import from all sources"""
    try:
        # Initialize data sources if not already done
        if not data_import_manager.data_sources:
            initialize_data_sources()

        # Perform import
        import_result = await data_import_manager.import_from_all_sources(days)

        return import_result

    except Exception as e:
        logger.error(f"Data import failed: {str(e)}")
        return {'error': str(e)}


def schedule_data_import():
    """Schedule regular data imports (would be used with Celery or similar)"""
    try:
        # This would be called by a task scheduler
        logger.info("Starting scheduled data import")

        # Import last 7 days of data
        result = asyncio.run(perform_data_import(7))

        logger.info(f"Scheduled import completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Scheduled data import failed: {str(e)}")
        return {'error': str(e)}


def import_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Import a specific PDF file"""
    try:
        # Process PDF
        processed_pdf = data_import_manager.pdf_processor.process_pdf(pdf_path)
        if not processed_pdf:
            return {'error': 'Failed to process PDF'}

        # Extract metadata
        metadata = data_import_manager.pdf_processor.extract_case_metadata(processed_pdf['text'])

        # Create case data
        case_data = {
            'title': metadata.get('title', f'Imported Case {datetime.now().isoformat()}'),
            'citation': metadata.get('citation', f'PDF Import {datetime.now().date()}'),
            'court': metadata.get('court', 'Unknown Court'),
            'judgment_date': metadata.get('judgment_date', datetime.now().date().isoformat()),
            'decision_date': datetime.now().date().isoformat(),
            'petitioners': 'Imported from PDF',
            'respondents': 'Imported from PDF',
            'case_text': processed_pdf['text'],
            'headnotes': processed_pdf['text'][:500] + '...' if len(processed_pdf['text']) > 500 else processed_pdf['text'],
            'case_type': 'judgment',
            'source': 'pdf_import',
            'source_id': f'pdf_{hashlib.md5(pdf_path.encode()).hexdigest()}',
            'source_url': pdf_path
        }

        # Process the case
        result = asyncio.run(data_import_manager.process_imported_cases([case_data], 'PDF Import'))

        return {
            'pdf_processed': True,
            'metadata_extracted': metadata,
            'import_result': result
        }

    except Exception as e:
        logger.error(f"PDF import failed: {str(e)}")
        return {'error': str(e)}