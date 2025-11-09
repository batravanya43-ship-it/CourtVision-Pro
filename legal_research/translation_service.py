"""
Translation Service for CourtVision Pro
Multilingual support for legal content with legal terminology preservation
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import re
import hashlib

from django.core.cache import cache
from django.conf import settings
from django.utils.translation import gettext as _

try:
    from googletrans import Translator
    GOOGLE_TRANSLATOR_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATOR_AVAILABLE = False

try:
    import indic_transliteration
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False

from .models import UserProfile, Case

logger = logging.getLogger(__name__)


class TranslationServiceError(Exception):
    """Custom exception for translation service errors"""
    pass


class LegalTerminologyManager:
    """Manages legal terminology translations and mappings"""

    def __init__(self):
        self.legal_term_mappings = {}
        self.legal_phrase_mappings = {}
        self.court_name_mappings = {}
        self.load_legal_terminology()

    def load_legal_terminology(self):
        """Load legal terminology mappings"""
        try:
            # English to Hindi legal terms
            self.legal_term_mappings = {
                'en_to_hi': {
                    'contract': 'अनुबंध',
                    'agreement': 'समझौता',
                    'breach': 'उल्लंघन',
                    'damages': 'हर्जाना',
                    'injunction': 'निषेधादेश',
                    'plaintiff': 'वादी',
                    'defendant': 'प्रतिवादी',
                    'petitioner': 'याचिकाकर्ता',
                    'respondent': 'उत्तरदाता',
                    'judgment': 'न्यायाधीश',
                    'decree': 'डिक्री',
                    'appeal': 'अपील',
                    'jurisdiction': 'अधिकार क्षेत्र',
                    'precedent': 'पूर्वनिर्णय',
                    'statute': 'अधिनियम',
                    'liable': 'उत्तरदायी',
                    'negligence': 'लापरवाही',
                    'fraud': 'धोखाधड़ी',
                    'evidence': 'सबूत',
                    'witness': 'गवाह',
                    'testimony': 'गवाही',
                    'verdict': 'फैसला',
                    'litigation': 'मुकदमेबाजी',
                    'settlement': 'समझौता',
                    'arbitration': 'पंचाट',
                    'mediation': 'मध्यस्थता',
                    'commercial dispute': 'व्यावसायिक विवाद',
                    'intellectual property': 'बौद्धिक संपदा',
                    'trademark': 'ट्रेडमार्क',
                    'copyright': 'कॉपीराइट',
                    'patent': 'पेटेंट'
                },
                'en_to_ta': {
                    'contract': 'ஒப்பந்தம்',
                    'agreement': 'ஒப்பந்தம்',
                    'breach': 'மீறல்',
                    'damages': 'இழப்பீடு',
                    'injunction': 'தடை உத்தரவு',
                    'plaintiff': 'வாதி',
                    'defendant': 'பிரதிவாதி',
                    'petitioner': 'மனுதாரர்',
                    'respondent': 'பதிலளிப்பவர்',
                    'judgment': 'தீர்ப்பு',
                    'decree': 'தீர்ப்பு',
                    'appeal': 'மேல்முறையீடு',
                    'jurisdiction': 'அதிகார வரம்பு',
                    'precedent': 'முன்னுதாரணம்',
                    'statute': 'சட்டம்',
                    'liable': 'பொறுப்புள்ள',
                    'negligence': 'அலட்சியம்',
                    'fraud': 'மோசடி',
                    'evidence': 'ஆதாரம்',
                    'witness': 'சாட்சி',
                    'testimony': 'சாட்சி',
                    'verdict': '�ீர்ப்பு',
                    'litigation': 'வழக்கு',
                    'settlement': 'ஒப்பந்தம்',
                    'arbitration': 'நடுவர் தீர்ப்பு',
                    'mediation': '�டைத்தரகம்',
                    'commercial dispute': 'வணிக தகராறு',
                    'intellectual property': 'அறிவசார் சொத்து',
                    'trademark': 'வர்த்தக முத்திரை',
                    'copyright': 'பதிப்புரிமை',
                    'patent': 'காப்புரிமம்'
                },
                'en_to_te': {
                    'contract': 'ఒప్పందం',
                    'agreement': 'అవగాహన',
                    'breach': 'ఉల్లంఘన',
                    'damages': 'నష్టపరిహారం',
                    'injunction': 'ఆజ్ఞాపలం',
                    'plaintiff': 'వాది',
                    'defendant': 'ప్రతివాది',
                    'petitioner': 'విన్నపుదారు',
                    'respondent': 'స్పందించేవారు',
                    'judgment': 'తీర్పు',
                    'decree': 'డిక్రీ',
                    'appeal': 'అప్పీల్',
                    'jurisdiction': 'అధికార పరిధి',
                    'precedent': 'అధ్యాయం',
                    'statute': 'చట్టం',
                    'liable': 'బాధ్యులు',
                    'negligence': 'నిర్లక్ష్యం',
                    'fraud': 'మోసం',
                    'evidence': 'ఆధారాలు',
                    'witness': 'సాక్షి',
                    'testimony': 'సాక్ష్యం',
                    'verdict': 'తీర్పు',
                    'litigation': 'వ్యాజ్యం',
                    'settlement': 'ఒప్పందం',
                    'arbitration': 'మధ్యవర్తిత్వం',
                    'mediation': '�ధ్యస్థత',
                    'commercial dispute': 'వాణిజ్య వివాదం',
                    'intellectual property': 'మేధో సంపద',
                    'trademark': 'వ్యాపార గుర్తు',
                    'copyright': 'కాపీరైట్',
                    'patent': 'పేటెంట్'
                }
            }

            # Legal phrase mappings
            self.legal_phrase_mappings = {
                'en_to_hi': {
                    'breach of contract': 'अनुबंध का उल्लंघन',
                    'terms and conditions': 'नियम और शर्तें',
                    'force majeure': 'अचानक घटना',
                    'good faith': 'ईमानदारी से',
                    'due diligence': 'उचित सावधानी',
                    'legal precedent': 'कानूनी पूर्वनिर्णय',
                    'case law': 'चले आ रहे कानून',
                    'statutory provisions': 'वैधानिक प्रावधान',
                    'intellectual property rights': 'बौद्धिक संपदा अधिकार',
                    'court of law': 'न्यायालय',
                    'high court': 'उच्च न्यायालय',
                    'supreme court': 'सर्वोच्च न्यायालय',
                    'commercial court': 'व्यावसायिक न्यायालय',
                    'civil suit': 'सिविल मुकदमा',
                    'criminal case': 'आपराधिक मामला',
                    'family court': 'पारिवारिक न्यायालय'
                },
                'en_to_ta': {
                    'breach of contract': 'ஒப்பந்த மீறல்',
                    'terms and conditions': 'விதிமுறைகள் மற்றும் நிபந்தனைகள்',
                    'force majeure': 'அதிகாரம் அதிகப்படியான நிகழ்வு',
                    'good faith': 'நல்நம்பிக்கையுடன்',
                    'due diligence': 'உரிய கவனம்',
                    'legal precedent': 'சட்ட முன்னுதாரணம்',
                    'case law': 'வழக்குச் சட்டம்',
                    'statutory provisions': 'சட்ட விதிகள்',
                    'intellectual property rights': 'அறிவார்ந்ாய சொத்து உரிமைகள்',
                    'court of law': 'நீதிமன்றம்',
                    'high court': 'உயர் நீதிமன்றம்',
                    'supreme court': 'உச்ச நீதிமன்றம்',
                    'commercial court': 'வணிக நீதிமன்றம்',
                    'civil suit': 'சிவில் வழக்கு',
                    'criminal case': 'குற்றவியல் வழக்கு',
                    'family court': 'குடுஂப நீதிமன்றம்'
                },
                'en_to_te': {
                    'breach of contract': 'ఒప్పందం ఉల్లంఘన',
                    'terms and conditions': 'నియమాలు మరియు షరతులు',
                    'force majeure': 'అసాధారణ సంఘటన',
                    'good faith': 'మంచి నమ్మకంతో',
                    'due diligence': 'సరైన శ్రద్ధ',
                    'legal precedent': 'చట్టపరమైన అధ్యాయం',
                    'case law': 'వ్యాజ్యం చట్టం',
                    'statutory provisions': 'చట్టపరమైన నిబంధనలు',
                    'intellectual property rights': 'మేధో సంపద హక్కులు',
                    'court of law': 'న్యాయస్థానం',
                    'high court': 'హైకోర్ట్',
                    'supreme court': 'సుప్రీమ్ కోర్ట్',
                    'commercial court': 'వాణిజ్య న్యాయస్థానం',
                    'civil suit': 'సివిల్ వ్యాజ్యం',
                    'criminal case': 'క్రిమినల్ కేసు',
                    'family court': 'కుటుంబ న్యాయస్థానం'
                }
            }

            # Court name mappings
            self.court_name_mappings = {
                'en_to_hi': {
                    'Supreme Court of India': 'भारत का सर्वोच्च न्यायालय',
                    'Delhi High Court': 'दिल्ली उच्च न्यायालय',
                    'Bombay High Court': 'बॉम्बे उच्च न्यायालय',
                    'Calcutta High Court': 'कलकत्ता उच्च न्यायालय',
                    'Madras High Court': 'मद्रास उच्च न्यायालय'
                },
                'en_to_ta': {
                    'Supreme Court of India': 'இந்திய உச்ச நீதிமன்றம்',
                    'Delhi High Court': 'தில்லி உயர் நீதிமன்றம்',
                    'Bombay High Court': 'பம்பாய் உயர் நீதிமன்றம்',
                    'Calcutta High Court': 'கல்கத்தா உயர் நீதிமன்றம்',
                    'Madras High Court': 'சென்னை உயர் நீதிமன்றம்'
                },
                'en_to_te': {
                    'Supreme Court of India': 'భారత సుప్రీమ్ కోర్ట్',
                    'Delhi High Court': 'ఢిల్లీ హైకోర్ట్',
                    'Bombay High Court': 'బాంబే హైకోర్ట్',
                    'Calcutta High Court': 'కలకత్తా హైకోర్ట్',
                    'Madras High Court': 'మద్రాస్ హైకోర్ట్'
                }
            }

            logger.info("Legal terminology mappings loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load legal terminology: {str(e)}")

    def translate_legal_term(self, term: str, source_lang: str, target_lang: str) -> str:
        """Translate a specific legal term"""
        try:
            mapping_key = f"{source_lang}_to_{target_lang}"
            term_mappings = self.legal_term_mappings.get(mapping_key, {})
            return term_mappings.get(term.lower(), term)
        except Exception as e:
            logger.error(f"Legal term translation failed: {str(e)}")
            return term

    def translate_legal_phrase(self, phrase: str, source_lang: str, target_lang: str) -> str:
        """Translate a legal phrase"""
        try:
            mapping_key = f"{source_lang}_to_{target_lang}"
            phrase_mappings = self.legal_phrase_mappings.get(mapping_key, {})
            return phrase_mappings.get(phrase.lower(), phrase)
        except Exception as e:
            logger.error(f"Legal phrase translation failed: {str(e)}")
            return phrase

    def translate_court_name(self, court_name: str, source_lang: str, target_lang: str) -> str:
        """Translate court name"""
        try:
            mapping_key = f"{source_lang}_to_{target_lang}"
            court_mappings = self.court_name_mappings.get(mapping_key, {})
            return court_mappings.get(court_name, court_name)
        except Exception as e:
            logger.error(f"Court name translation failed: {str(e)}")
            return court_name


class MultilingualSearchProcessor:
    """Processes multilingual search queries and results"""

    def __init__(self):
        self.terminology_manager = LegalTerminologyManager()
        self.translator = None
        self.initialize_translator()

    def initialize_translator(self):
        """Initialize translation service"""
        try:
            if GOOGLE_TRANSLATOR_AVAILABLE:
                self.translator = Translator()
                logger.info("Google Translator initialized")
            else:
                logger.warning("Google Translator not available")
        except Exception as e:
            logger.error(f"Translator initialization failed: {str(e)}")

    async def process_search_query(self, query: str, source_lang: str = 'en',
                                 target_lang: str = 'en') -> Dict[str, Any]:
        """Process and translate search query"""
        try:
            processed_query = {
                'original_query': query,
                'source_language': source_lang,
                'target_language': target_lang,
                'processed_terms': [],
                'translated_query': query,
                'legal_terms_detected': [],
                'search_suggestions': []
            }

            # Detect and translate legal terms
            legal_terms = self._extract_legal_terms(query, source_lang)
            processed_query['legal_terms_detected'] = legal_terms

            # Translate query if needed
            if source_lang != target_lang:
                translated_query = await self._translate_text(query, source_lang, target_lang)
                processed_query['translated_query'] = translated_query

                # Apply legal terminology corrections
                corrected_query = self._apply_legal_terminology_corrections(
                    translated_query, source_lang, target_lang
                )
                processed_query['corrected_query'] = corrected_query
            else:
                processed_query['corrected_query'] = query

            # Generate search suggestions
            suggestions = self._generate_search_suggestions(query, source_lang, target_lang)
            processed_query['search_suggestions'] = suggestions

            return processed_query

        except Exception as e:
            logger.error(f"Search query processing failed: {str(e)}")
            return {
                'original_query': query,
                'error': str(e),
                'processed_query': query
            }

    def _extract_legal_terms(self, text: str, language: str) -> List[str]:
        """Extract legal terms from text"""
        try:
            if language == 'en':
                # Simple keyword extraction for English
                legal_keywords = [
                    'contract', 'agreement', 'breach', 'damages', 'injunction',
                    'plaintiff', 'defendant', 'petitioner', 'respondent',
                    'judgment', 'appeal', 'jurisdiction', 'precedent',
                    'statute', 'liable', 'negligence', 'fraud', 'evidence',
                    'commercial dispute', 'intellectual property', 'trademark',
                    'copyright', 'patent', 'arbitration', 'mediation'
                ]

                found_terms = []
                text_lower = text.lower()
                for term in legal_keywords:
                    if term in text_lower:
                        found_terms.append(term)

                return found_terms

            return []

        except Exception as e:
            logger.error(f"Legal term extraction failed: {str(e)}")
            return []

    async def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using available translation service"""
        try:
            if self.translator and GOOGLE_TRANSLATOR_AVAILABLE:
                # Check cache first
                cache_key = f"translation_{hashlib.md5(f'{text}_{source_lang}_{target_lang}'.encode()).hexdigest()}"
                cached_translation = cache.get(cache_key)
                if cached_translation:
                    return cached_translation

                # Perform translation
                result = self.translator.translate(text, src=source_lang, dest=target_lang)
                translation = result.text

                # Cache result
                cache.set(cache_key, translation, timeout=3600)  # 1 hour

                return translation

            # Fallback to terminology-based translation
            return self._terminology_based_translation(text, source_lang, target_lang)

        except Exception as e:
            logger.error(f"Text translation failed: {str(e)}")
            return text

    def _terminology_based_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """Fallback translation using legal terminology mappings"""
        try:
            translated_text = text

            # Translate legal terms
            legal_terms = self._extract_legal_terms(text, source_lang)
            for term in legal_terms:
                translated_term = self.terminology_manager.translate_legal_term(
                    term, source_lang, target_lang
                )
                translated_text = translated_text.replace(term, translated_term)

            # Translate legal phrases
            for phrase, translation in self.terminology_manager.legal_phrase_mappings.get(
                f"{source_lang}_to_{target_lang}", {}
            ).items():
                translated_text = translated_text.replace(phrase, translation)

            return translated_text

        except Exception as e:
            logger.error(f"Terminology-based translation failed: {str(e)}")
            return text

    def _apply_legal_terminology_corrections(self, text: str, source_lang: str, target_lang: str) -> str:
        """Apply legal terminology corrections to translated text"""
        try:
            corrected_text = text

            # Replace common mistranslations
            corrections = self._get_translation_corrections(source_lang, target_lang)
            for incorrect, correct in corrections.items():
                corrected_text = corrected_text.replace(incorrect, correct)

            return corrected_text

        except Exception as e:
            logger.error(f"Legal terminology corrections failed: {str(e)}")
            return text

    def _get_translation_corrections(self, source_lang: str, target_lang: str) -> Dict[str, str]:
        """Get common translation corrections"""
        corrections = {}

        if source_lang == 'en' and target_lang == 'hi':
            corrections.update({
                'न्यायाधीश': 'फैसला',  # Judge -> Judgment
                'अदालत': 'न्यायालय',  # Court -> Court
                'कानून': 'अधिनियम',  # Law -> Act
            })
        elif source_lang == 'en' and target_lang == 'ta':
            corrections.update({
                'நீதிபதி': 'தீர்ப்பு',  # Judge -> Judgment
                'நீதிமன்றம்': 'நீதிமன்றம்',  # Court -> Court
                'சட்டம்': 'சட்டம்',  # Law -> Act
            })
        elif source_lang == 'en' and target_lang == 'te':
            corrections.update({
                'న్యాయమూర్తి': 'తీర్పు',  # Judge -> Judgment
                'కోర్టు': 'న్యాయస్థానం',  # Court -> Court
                'చట్టం': 'చట్టం',  # Law -> Act
            })

        return corrections

    def _generate_search_suggestions(self, query: str, source_lang: str, target_lang: str) -> List[str]:
        """Generate search suggestions"""
        suggestions = []

        try:
            # Extract legal terms and suggest related terms
            legal_terms = self._extract_legal_terms(query, source_lang)

            for term in legal_terms:
                # Suggest synonyms
                synonyms = self._get_legal_synonyms(term, source_lang)
                for synonym in synonyms[:2]:  # Limit suggestions
                    if synonym not in query.lower():
                        suggested_query = query.replace(term, synonym, 1)
                        suggestions.append(suggested_query)

                # Suggest related legal concepts
                related_concepts = self._get_related_concepts(term, source_lang)
                for concept in related_concepts[:1]:  # Limit suggestions
                    suggested_query = f"{query} {concept}"
                    suggestions.append(suggested_query)

        except Exception as e:
            logger.error(f"Search suggestion generation failed: {str(e)}")

        return suggestions[:5]  # Return top 5 suggestions

    def _get_legal_synonyms(self, term: str, language: str) -> List[str]:
        """Get legal synonyms for a term"""
        synonyms = {
            'contract': ['agreement', 'pact', 'compact'],
            'breach': ['violation', 'infraction', 'contravention'],
            'damages': ['compensation', 'reparation', 'indemnity'],
            'injunction': ['order', 'decree', 'directive'],
            'liable': ['responsible', 'accountable', 'obligated']
        }

        return synonyms.get(term.lower(), [])

    def _get_related_concepts(self, term: str, language: str) -> List[str]:
        """Get related legal concepts"""
        related = {
            'contract': ['terms and conditions', 'breach of contract', 'force majeure'],
            'damages': ['compensation', 'liquidated damages', 'punitive damages'],
            'injunction': ['temporary injunction', 'permanent injunction', 'mandatory injunction'],
            'arbitration': ['arbitration clause', 'arbitral award', 'institutional arbitration'],
            'intellectual property': ['trademark', 'copyright', 'patent', 'trade secret']
        }

        return related.get(term.lower(), [])

    async def translate_search_results(self, results: List[Dict], target_lang: str = 'en') -> List[Dict]:
        """Translate search results to target language"""
        try:
            translated_results = []

            for result in results:
                translated_result = result.copy()

                # Translate key fields
                if target_lang != 'en':
                    # Translate title
                    if result.get('title'):
                        translated_title = await self._translate_text(
                            result['title'], 'en', target_lang
                        )
                        translated_result['title_translated'] = translated_title

                    # Translate snippet
                    if result.get('snippet'):
                        translated_snippet = await self._translate_text(
                            result['snippet'], 'en', target_lang
                        )
                        translated_result['snippet_translated'] = translated_snippet

                    # Translate court name
                    if result.get('court'):
                        translated_court = self.terminology_manager.translate_court_name(
                            result['court'], 'en', target_lang
                        )
                        translated_result['court_translated'] = translated_court

                    # Translate legal highlights
                    if result.get('highlights'):
                        translated_highlights = []
                        for highlight in result['highlights']:
                            translated_highlight = await self._translate_text(
                                highlight, 'en', target_lang
                            )
                            translated_highlights.append(translated_highlight)
                        translated_result['highlights_translated'] = translated_highlights

                translated_results.append(translated_result)

            return translated_results

        except Exception as e:
            logger.error(f"Search results translation failed: {str(e)}")
            return results


class LanguageDetectionService:
    """Service for detecting languages in text"""

    def __init__(self):
        self.language_patterns = {
            'hi': r'[\u0900-\u097F]',  # Devanagari script
            'ta': r'[\u0B80-\u0BFF]',  # Tamil script
            'te': r'[\u0C00-\u0C7F]',  # Telugu script
        }

    def detect_language(self, text: str) -> str:
        """Detect the primary language of text"""
        try:
            if not text or len(text.strip()) == 0:
                return 'en'

            # Check for Indic language scripts
            for lang_code, pattern in self.language_patterns.items():
                if re.search(pattern, text):
                    return lang_code

            # Default to English
            return 'en'

        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return 'en'

    def is_multilingual(self, text: str) -> bool:
        """Check if text contains multiple languages"""
        try:
            detected_languages = set()

            for lang_code, pattern in self.language_patterns.items():
                if re.search(pattern, text):
                    detected_languages.add(lang_code)

            # Check if English is also present
            if re.search(r'[a-zA-Z]', text):
                detected_languages.add('en')

            return len(detected_languages) > 1

        except Exception as e:
            logger.error(f"Multilingual detection failed: {str(e)}")
            return False


# Global instances
legal_terminology_manager = LegalTerminologyManager()
multilingual_processor = MultilingualSearchProcessor()
language_detector = LanguageDetectionService()


async def process_multilingual_search(query: str, user_language: str = 'en') -> Dict[str, Any]:
    """Process multilingual search query and return results"""
    try:
        # Detect query language
        detected_lang = language_detector.detect_language(query)

        # Process search query
        processed_query = await multilingual_processor.process_search_query(
            query, detected_lang, 'en'  # Always search in English
        )

        # Get search results (this would integrate with the search engine)
        # For now, return processed query info
        return {
            'processed_query': processed_query,
            'detected_language': detected_lang,
            'user_language': user_language,
            'needs_translation': detected_lang != 'en',
            'multilingual_features': {
                'legal_terms_detected': len(processed_query.get('legal_terms_detected', [])),
                'search_suggestions': len(processed_query.get('search_suggestions', [])),
                'corrections_applied': 'corrected_query' in processed_query
            }
        }

    except Exception as e:
        logger.error(f"Multilingual search processing failed: {str(e)}")
        return {
            'error': str(e),
            'original_query': query,
            'detected_language': 'en'
        }


def translate_legal_content(content: str, target_language: str, source_language: str = 'en') -> Dict[str, Any]:
    """Translate legal content with terminology preservation"""
    try:
        if source_language == target_language:
            return {
                'translated_content': content,
                'source_language': source_language,
                'target_language': target_language,
                'legal_terms_preserved': True,
                'translation_method': 'none'
            }

        # Apply terminology-based translation
        translated_content = multilingual_processor._terminology_based_translation(
            content, source_language, target_language
        )

        # Detect legal terms that were translated
        source_terms = multilingual_processor._extract_legal_terms(content, source_language)
        target_terms = multilingual_processor._extract_legal_terms(translated_content, target_language)

        return {
            'translated_content': translated_content,
            'source_language': source_language,
            'target_language': target_language,
            'legal_terms_preserved': len(source_terms) > 0,
            'legal_terms_count': len(source_terms),
            'translation_method': 'terminology_based',
            'source_legal_terms': source_terms,
            'target_legal_terms': target_terms
        }

    except Exception as e:
        logger.error(f"Legal content translation failed: {str(e)}")
        return {
            'error': str(e),
            'original_content': content,
            'source_language': source_language,
            'target_language': target_language
        }