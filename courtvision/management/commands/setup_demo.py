from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from legal_research.models import HighCourt, UserProfile, Suit, Tag, Case
from decimal import Decimal
from datetime import datetime, timedelta
import uuid
import random


class Command(BaseCommand):
    help = 'Set up demo data for CourtVision Pro'

    def handle(self, *args, **options):
        self.stdout.write('Setting up demo data for CourtVision Pro...')

        # Create demo superuser
        if not User.objects.filter(username='admin').exists():
            admin_user = User.objects.create_superuser(
                username='admin',
                email='admin@courtvision.com',
                password='admin123',
                first_name='Demo',
                last_name='Administrator'
            )
            self.stdout.write(self.style.SUCCESS('Created admin user: admin/admin123'))
        else:
            admin_user = User.objects.get(username='admin')
            self.stdout.write('Admin user already exists')

        # Create High Courts
        courts_data = [
            ('Delhi High Court', 'Delhi', 'DEL'),
            ('Mumbai High Court', 'Maharashtra', 'BOM'),
            ('Chennai High Court', 'Tamil Nadu', 'MAD'),
            ('Kolkata High Court', 'West Bengal', 'CAL'),
        ]

        for name, jurisdiction, code in courts_data:
            court, created = HighCourt.objects.get_or_create(
                code=code,
                defaults={
                    'name': name,
                    'jurisdiction': jurisdiction,
                    'established_date': datetime(1862, 1, 1).date(),
                }
            )
            if created:
                self.stdout.write(f'Created High Court: {name}')

        # Create User Profile for admin
        if not hasattr(admin_user, 'userprofile'):
            delhi_court = HighCourt.objects.get(code='DEL')
            UserProfile.objects.create(
                user=admin_user,
                high_court=delhi_court,
                designation='Senior Judicial Officer',
                employee_id='JD001',
                phone_number='+91-9876543210',
                default_language='en',
                notification_settings={
                    'email_notifications': True,
                    'search_alerts': True,
                    'case_updates': False,
                },
                preferences={
                    'default_search_scope': 'all_courts',
                    'results_per_page': 20,
                    'auto_save_searches': True,
                }
            )
            self.stdout.write('Created user profile for admin')

        # Create Tags
        tags_data = [
            ('Contract Law', 'Legal principles related to contracts', '#007bff'),
            ('Property Law', 'Real estate and property disputes', '#28a745'),
            ('Commercial Law', 'Business and commercial disputes', '#ffc107'),
            ('Family Law', 'Family and marriage related matters', '#dc3545'),
            ('Criminal Law', 'Criminal proceedings and offenses', '#6f42c1'),
            ('Constitutional Law', 'Constitutional interpretation and rights', '#fd7e14'),
            ('Tax Law', 'Taxation and revenue matters', '#20c997'),
            ('Labor Law', 'Employment and labor relations', '#e83e8c'),
        ]

        for name, description, color in tags_data:
            tag, created = Tag.objects.get_or_create(
                name=name,
                defaults={
                    'description': description,
                    'color': color,
                }
            )
            if created:
                self.stdout.write(f'Created tag: {name}')

        # Create Demo Suits
        suits_data = [
            ('Contract Dispute Resolution', 'Ongoing contract dispute between commercial parties', 'commercial', 'high'),
            ('Property Title Verification', 'Land title verification and clarification suit', 'civil', 'medium'),
            ('Commercial Arbitration', 'International commercial arbitration proceedings', 'commercial', 'urgent'),
        ]

        for name, description, suit_type, priority in suits_data:
            suit, created = Suit.objects.get_or_create(
                name=name,
                defaults={
                    'description': description,
                    'suit_type': suit_type,
                    'priority_level': priority,
                    'created_by': admin_user.userprofile,
                }
            )
            if created:
                suit.assigned_users.add(admin_user.userprofile)
                self.stdout.write(f'Created suit: {name}')

        # Create Demo Cases
        self.create_demo_cases()

        # Create Search History
        self.create_demo_search_history(admin_user)

        self.stdout.write(self.style.SUCCESS('Demo data setup completed!'))
        self.stdout.write('Login with: admin / admin123')

    def create_demo_cases(self):
        """Create demo legal cases"""
        tags = list(Tag.objects.all())
        courts = list(HighCourt.objects.all())

        case_templates = [
            {
                'title': 'Breach of Contract - Delivery Non-Performance',
                'citation': '2023 SCC Online Del 1234',
                'summary': 'Case concerning breach of supply contract where supplier failed to deliver goods within stipulated time period.',
                'keywords': ['contract', 'breach', 'supply', 'commercial']
            },
            {
                'title': 'Title Dispute - Inherited Property',
                'citation': '2023 SCC Online Bom 5678',
                'summary': 'Property title dispute regarding inheritance of ancestral property among siblings.',
                'keywords': ['property', 'inheritance', 'title', 'family']
            },
            {
                'title': 'Commercial Arbitration Award Enforcement',
                'citation': '2023 SCC Online Mad 9012',
                'summary': 'Petition for enforcement of international commercial arbitration award.',
                'keywords': ['arbitration', 'international', 'commercial', 'enforcement']
            },
            {
                'title': 'Intellectual Property Infringement',
                'citation': '2023 SCC Online Del 3456',
                'summary': 'Trademark infringement case involving unauthorized use of registered brand.',
                'keywords': ['trademark', 'intellectual property', 'infringement']
            },
            {
                'title': 'Employment Termination Dispute',
                'citation': '2023 SCC Online Cal 7890',
                'summary': 'Wrongful termination of employment and compensation claim.',
                'keywords': ['employment', 'termination', 'labor', 'compensation']
            }
        ]

        for template in case_templates:
            case_id = uuid.uuid4()
            court = random.choice(courts)
            case_tags = random.sample(tags, random.randint(1, 3))

            # Generate random judgment date within last 2 years
            judgment_date = datetime.now() - timedelta(days=random.randint(1, 730))

            case, created = Case.objects.get_or_create(
                id=case_id,
                defaults={
                    'title': template['title'],
                    'citation': template['citation'],
                    'court': court,
                    'bench': f"Hon'ble Justices J. Smith & J. Kumar",
                    'judgment_date': judgment_date.date(),
                    'decision_date': judgment_date.date(),
                    'petitioners': f"M/s {template['title'].split(' - ')[0]}",
                    'respondents': f"M/s {template['title'].split(' - ')[1] if ' - ' in template['title'] else 'Opposition Party'}",
                    'case_text': f"This is the full text of the judgment for {template['title']}. " +
                                "The court considered various legal precedents and statutory provisions. " +
                                "After careful consideration of arguments from both sides, the court delivered its judgment based on established legal principles. " +
                                "The judgment includes detailed analysis of applicable laws and precedents.",
                    'headnotes': f"Key legal points from {template['title']}. Court analyzed contract provisions and applicable statutory framework.",
                    'ai_summary': {
                        'summary': f"AI-generated summary of {template['title']}. The court examined the contractual obligations and statutory provisions applicable to the case.",
                        'key_points': [
                            "Contractual obligations must be performed in good faith",
                            "Non-performance may constitute breach of contract",
                            "Damages awarded must be reasonable and proportionate"
                        ],
                        'decision': f"Judgment delivered in favor of petitioner with compensation awarded",
                        'implications': f"This case establishes important precedent for {template['keywords'][0]} matters"
                    },
                    'extracted_principles': [
                        f"Legal principle 1 related to {template['keywords'][0]}",
                        f"Legal principle 2 from statutory interpretation",
                        f"Legal principle 3 regarding remedy and relief"
                    ],
                    'statutes_cited': [
                        "Indian Contract Act, 1872",
                        "Specific Relief Act, 1963",
                        "Code of Civil Procedure, 1908"
                    ],
                    'precedents_cited': [
                        "2022 SCC 123 - Leading case on similar matter",
                        "2021 SCC 456 - Established legal principle",
                        "2020 SCC 789 - Landmark judgment"
                    ],
                    'case_type': random.choice(['judgment', 'order', 'appeal']),
                    'relevance_score': random.uniform(60, 95),
                    'is_published': True,
                    'view_count': random.randint(10, 500)
                }
            )

            if created:
                case.tags.set(case_tags)
                self.stdout.write(f'Created case: {template["title"]}')

    def create_demo_search_history(self, user):
        """Create demo search history"""
        search_queries = [
            'breach of contract',
            'property title dispute',
            'commercial arbitration',
            'intellectual property',
            'employment law',
            'specific performance',
            'damages calculation',
            'legal precedent'
        ]

        for query in search_queries:
            # Create search with random timestamp in last 30 days
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            SearchHistory.objects.get_or_create(
                user=user,
                query_text=query,
                timestamp=timestamp,
                defaults={
                    'filters': {},
                    'results_count': random.randint(5, 50),
                    'search_time': round(random.uniform(0.5, 2.0), 2)
                }
            )