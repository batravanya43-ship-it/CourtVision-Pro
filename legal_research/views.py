from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.core.serializers.json import DjangoJSONEncoder
from django.utils import translation
from django.utils.translation import gettext as _
from django.db.models import Q, Count, Avg
from django.urls import reverse
from django.utils import timezone
from django.conf import settings
from django.template.loader import render_to_string
import json
import uuid
import csv
import io
from datetime import datetime, timedelta
import random

from .models import (
    HighCourt, UserProfile, Suit, Tag, Case, SearchHistory,
    Customization, UserNote, SavedCase, AnalyticsData
)


def landing_page(request):
    """Landing page view"""
    if request.user.is_authenticated:
        return redirect('legal_research:dashboard')

    return render(request, 'legal_research/landing.html')


@login_required
def dashboard(request):
    """Main dashboard view"""
    user = request.user
    user_profile = getattr(user, 'userprofile', None)

    # Get recent searches
    recent_searches = SearchHistory.objects.filter(user=user).order_by('-timestamp')[:5]

    # Get statistics
    total_searches = SearchHistory.objects.filter(user=user).count()
    saved_cases = SavedCase.objects.filter(user=user).count()
    user_notes = UserNote.objects.filter(user=user).count()

    # Get active suits if user has profile
    active_suits = []
    if user_profile:
        active_suits = user_profile.get_active_suits()

    # System status (mock data for now)
    system_status = {
        'search_engine': 'operational',
        'ai_service': 'operational',
        'database': 'operational',
        'last_updated': timezone.now(),
        'uptime': '99.9%'
    }

    # Quick links configuration
    quick_links = [
        {'title': _('Advanced Search'), 'url': reverse('legal_research:search'), 'icon': 'bi-search', 'color': 'primary'},
        {'title': _('Analytics'), 'url': reverse('legal_research:analytics'), 'icon': 'bi-graph-up', 'color': 'success'},
        {'title': _('Customization'), 'url': reverse('legal_research:customization'), 'icon': 'bi-sliders', 'color': 'info'},
        {'title': _('Upload Data'), 'url': reverse('legal_research:upload'), 'icon': 'bi-upload', 'color': 'warning'},
    ]

    context = {
        'user_profile': user_profile,
        'recent_searches': recent_searches,
        'total_searches': total_searches,
        'saved_cases': saved_cases,
        'user_notes': user_notes,
        'active_suits': active_suits,
        'system_status': system_status,
        'quick_links': quick_links,
    }

    return render(request, 'legal_research/dashboard.html', context)


@login_required
def advanced_search(request):
    """Advanced search page"""
    user = request.user
    user_profile = getattr(user, 'userprofile', None)

    # Get filter options
    high_courts = HighCourt.objects.filter(is_active=True).order_by('name')
    tags = Tag.objects.all().order_by('name')
    case_types = Case.CASE_TYPES if hasattr(Case, 'CASE_TYPES') else [
        ('judgment', 'Judgment'),
        ('order', 'Order'),
        ('appeal', 'Appeal'),
    ]

    # Get user's suits if available
    user_suits = []
    if user_profile:
        user_suits = user_profile.get_active_suits()

    # Get recent searches for suggestions
    recent_searches = SearchHistory.objects.filter(user=user).order_by('-timestamp')[:10]

    context = {
        'high_courts': high_courts,
        'tags': tags,
        'case_types': case_types,
        'user_suits': user_suits,
        'recent_searches': recent_searches,
        'user_profile': user_profile,
    }

    return render(request, 'legal_research/search/search.html', context)


@login_required
def search_results(request):
    """Handle search requests"""
    if request.method != 'POST':
        return redirect('legal_research:search')

    query = request.POST.get('query', '').strip()
    filters = {}

    # Extract filters from form data
    for key, value in request.POST.items():
        if key.startswith('filter_') and value:
            filter_key = key.replace('filter_', '')
            filters[filter_key] = value

    if not query:
        messages.warning(request, _('Please enter a search query'))
        return redirect('legal_research:search')

    # Save search to history
    SearchHistory.objects.create(
        user=request.user,
        query_text=query,
        filters=filters,
        results_count=0,  # Will be updated later
        search_time=0.0   # Will be calculated later
    )

    # Perform search (mock implementation for now)
    results = perform_mock_search(query, filters)

    context = {
        'query': query,
        'results': results,
        'filters': filters,
        'total_results': len(results),
    }

    return render(request, 'legal_research/search/results.html', context)


@csrf_exempt
@require_POST
def ajax_search_suggestions(request):
    """AJAX endpoint for search suggestions"""
    query = request.POST.get('query', '').strip()

    if len(query) < 2:
        return JsonResponse({'suggestions': []})

    # Mock suggestions - in production, this would query the database
    suggestions = generate_mock_suggestions(query)

    return JsonResponse({'suggestions': suggestions})


@csrf_exempt
@require_POST
def ajax_search_results(request):
    """AJAX endpoint for search results"""
    query = request.POST.get('query', '').strip()
    filters = json.loads(request.POST.get('filters', '{}'))
    page = int(request.POST.get('page', 1))

    if not query:
        return JsonResponse({'error': 'Query is required'}, status=400)

    # Perform search
    results = perform_mock_search(query, filters)

    # Pagination
    items_per_page = 10
    paginator = Paginator(results, items_per_page)
    page_obj = paginator.get_page(page)

    # Prepare response
    response_data = {
        'results': [
            {
                'id': str(result['id']),
                'title': result['title'],
                'summary': result['summary'],
                'citation': result['citation'],
                'court': result['court'],
                'judgment_date': result['judgment_date'],
                'tags': result['tags'],
                'relevance_score': result['relevance_score'],
            }
            for result in page_obj.object_list
        ],
        'pagination': {
            'current_page': page_obj.number,
            'total_pages': paginator.num_pages,
            'has_previous': page_obj.has_previous(),
            'has_next': page_obj.has_next(),
            'total_items': paginator.count,
        }
    }

    return JsonResponse(response_data)


@login_required
def case_detail(request, case_id):
    """Case detail view"""
    try:
        case = Case.objects.get(id=case_id)
    except Case.DoesNotExist:
        raise Http404(_("Case not found"))

    # Increment view count
    case.view_count += 1
    case.save(update_fields=['view_count'])

    # Get user's note for this case
    user_note = None
    if request.user.is_authenticated:
        try:
            user_note = UserNote.objects.get(user=request.user, case=case)
        except UserNote.DoesNotExist:
            pass

    # Get related cases
    related_cases = case.get_related_cases(limit=5)

    # Check if case is saved by user
    is_saved = False
    if request.user.is_authenticated:
        is_saved = SavedCase.objects.filter(user=request.user, case=case).exists()

    context = {
        'case': case,
        'user_note': user_note,
        'related_cases': related_cases,
        'is_saved': is_saved,
    }

    return render(request, 'legal_research/cases/case_detail.html', context)


@login_required
def save_case(request, case_id):
    """Save a case for later reference"""
    case = get_object_or_404(Case, id=case_id)

    saved_case, created = SavedCase.objects.get_or_create(
        user=request.user,
        case=case,
        defaults={'folder': 'General'}
    )

    if created:
        messages.success(request, _('Case saved successfully'))
    else:
        messages.info(request, _('Case was already saved'))

    return redirect('legal_research:case_detail', case_id=case_id)


@login_required
def export_case(request, case_id):
    """Export case data"""
    case = get_object_or_404(Case, id=case_id)
    format_type = request.GET.get('format', 'pdf')

    if format_type == 'pdf':
        # Generate PDF (mock implementation)
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{case.citation}.pdf"'
        # PDF generation would go here
        response.write(b'%PDF-1.4 mock PDF content')
        return response

    elif format_type == 'txt':
        # Generate text file
        response = HttpResponse(content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename="{case.citation}.txt"'
        response.write(f"Title: {case.title}\n")
        response.write(f"Citation: {case.citation}\n")
        response.write(f"Court: {case.court}\n")
        response.write(f"Date: {case.judgment_date}\n\n")
        response.write(case.case_text)
        return response

    else:
        messages.error(request, _('Invalid export format'))
        return redirect('legal_research:case_detail', case_id=case_id)


@csrf_exempt
@require_POST
def save_case_note(request, case_id):
    """Save or update user note for a case"""
    case = get_object_or_404(Case, id=case_id)
    note_text = request.POST.get('note_text', '').strip()

    if not note_text:
        # Delete note if empty
        UserNote.objects.filter(user=request.user, case=case).delete()
        return JsonResponse({'success': True, 'message': _('Note deleted')})

    # Create or update note
    note, created = UserNote.objects.update_or_create(
        user=request.user,
        case=case,
        defaults={'note_text': note_text}
    )

    return JsonResponse({
        'success': True,
        'message': _('Note saved successfully'),
        'created': created
    })


@login_required
def customization_panel(request):
    """Customization panel view"""
    user = request.user
    user_profile = getattr(user, 'userprofile', None)

    if not user_profile:
        messages.error(request, _('Please complete your profile first'))
        return redirect('legal_research:profile')

    # Get user's suits
    user_suits = user_profile.get_active_suits()

    # Get existing customizations
    customizations = {}
    for suit in user_suits:
        try:
            customization = Customization.objects.get(user=user, suit=suit)
            customizations[suit.id] = customization
        except Customization.DoesNotExist:
            customizations[suit.id] = None

    context = {
        'user_suits': user_suits,
        'customizations': customizations,
        'user_profile': user_profile,
    }

    return render(request, 'legal_research/customization/customization_panel.html', context)


@csrf_exempt
@require_POST
def update_customization(request):
    """Update user customization preferences"""
    try:
        data = json.loads(request.body)
        suit_id = data.get('suit_id')

        if not suit_id:
            return JsonResponse({'error': 'Suit ID is required'}, status=400)

        suit = get_object_or_404(Suit, id=suit_id)

        # Validate that user has access to this suit
        user_profile = getattr(request.user, 'userprofile', None)
        if not user_profile or suit not in user_profile.get_active_suits():
            return JsonResponse({'error': 'Access denied'}, status=403)

        # Update or create customization
        customization, created = Customization.objects.update_or_create(
            user=request.user,
            suit=suit,
            defaults={
                'jurisdiction_emphasis': data.get('jurisdiction_emphasis', {}),
                'language_preferences': data.get('language_preferences', []),
                'precedent_statute_weight': float(data.get('precedent_statute_weight', 0.5)),
                'time_period_focus': data.get('time_period_focus', 'recent'),
                'analysis_focus_areas': data.get('analysis_focus_areas', []),
            }
        )

        return JsonResponse({
            'success': True,
            'message': _('Preferences saved successfully'),
            'created': created
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def load_customization(request):
    """Load customization preferences for a suit"""
    suit_id = request.GET.get('suit_id')
    if not suit_id:
        return JsonResponse({'error': 'Suit ID is required'}, status=400)

    try:
        suit = get_object_or_404(Suit, id=suit_id)
        customization = Customization.objects.get(user=request.user, suit=suit)

        return JsonResponse({
            'success': True,
            'preferences': {
                'jurisdiction_emphasis': customization.jurisdiction_emphasis,
                'language_preferences': customization.language_preferences,
                'precedent_statute_weight': customization.precedent_statute_weight,
                'time_period_focus': customization.time_period_focus,
                'analysis_focus_areas': customization.analysis_focus_areas,
            }
        })

    except Customization.DoesNotExist:
        return JsonResponse({
            'success': True,
            'preferences': {
                'jurisdiction_emphasis': {},
                'language_preferences': [],
                'precedent_statute_weight': 0.5,
                'time_period_focus': 'recent',
                'analysis_focus_areas': [],
            }
        })


@login_required
def analytics_dashboard(request):
    """Analytics dashboard view"""
    user = request.user

    # Generate mock analytics data
    analytics_data = generate_mock_analytics(user)

    context = {
        'analytics_data': analytics_data,
    }

    return render(request, 'legal_research/analytics/analytics_dashboard.html', context)


@csrf_exempt
def analytics_data(request):
    """AJAX endpoint for analytics data"""
    user = request.user
    period = request.GET.get('period', 'month')

    # Generate mock data based on period
    analytics_data = generate_mock_analytics(user, period)

    return JsonResponse({'data': analytics_data})


@login_required
def data_upload(request):
    """Data upload interface"""
    if request.method == 'POST':
        return handle_file_upload(request)

    return render(request, 'legal_research/upload/upload.html')


@csrf_exempt
def process_upload(request):
    """Process uploaded files"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    if not request.FILES.get('file'):
        return JsonResponse({'error': 'No file provided'}, status=400)

    uploaded_file = request.FILES['file']

    # Validate file type
    allowed_types = ['application/pdf', 'text/plain', 'application/json',
                    'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']

    if uploaded_file.content_type not in allowed_types:
        return JsonResponse({'error': 'Invalid file type'}, status=400)

    try:
        # Process file (mock implementation)
        result = process_uploaded_file(uploaded_file, request.user)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def user_profile(request):
    """User profile and settings page"""
    user = request.user
    user_profile = getattr(user, 'userprofile', None)

    if request.method == 'POST':
        return update_user_profile(request, user, user_profile)

    context = {
        'user_profile': user_profile,
        'high_courts': HighCourt.objects.filter(is_active=True).order_by('name'),
    }

    return render(request, 'legal_research/profile/profile.html', context)


def update_user_profile(request, user, user_profile):
    """Update user profile information"""
    try:
        # Update user basic info
        user.first_name = request.POST.get('first_name', '')
        user.last_name = request.POST.get('last_name', '')
        user.email = request.POST.get('email', '')
        user.save()

        # Update or create user profile
        if not user_profile:
            user_profile = UserProfile.objects.create(user=user)

        user_profile.designation = request.POST.get('designation', '')
        user_profile.employee_id = request.POST.get('employee_id', '')
        user_profile.phone_number = request.POST.get('phone_number', '')
        user_profile.default_language = request.POST.get('default_language', 'en')

        court_id = request.POST.get('high_court')
        if court_id:
            try:
                court = HighCourt.objects.get(id=court_id)
                user_profile.high_court = court
            except HighCourt.DoesNotExist:
                pass

        user_profile.save()

        messages.success(request, _('Profile updated successfully'))
        return redirect('legal_research:profile')

    except Exception as e:
        messages.error(request, _('Error updating profile: {}').format(str(e)))
        return redirect('legal_research:profile')


def help_page(request):
    """Help and ethical guidelines page"""
    return render(request, 'legal_research/help/help.html')


# API Endpoints
@csrf_exempt
@require_POST
def api_update_preferences(request):
    """API endpoint to update user preferences"""
    return update_customization(request)


@csrf_exempt
def api_load_preferences(request):
    """API endpoint to load user preferences"""
    return load_customization(request)


@csrf_exempt
@require_POST
def api_save_note(request):
    """API endpoint to save case notes"""
    case_id = json.loads(request.body).get('case_id')
    note_text = json.loads(request.body).get('note_text', '')
    return save_case_note(request, case_id)


@csrf_exempt
@require_POST
def api_save_case(request):
    """API endpoint to save cases"""
    try:
        data = json.loads(request.body)
        case_id = data.get('case_id')
        folder = data.get('folder', 'General')

        case = get_object_or_404(Case, id=case_id)
        saved_case, created = SavedCase.objects.get_or_create(
            user=request.user,
            case=case,
            defaults={'folder': folder}
        )

        return JsonResponse({
            'success': True,
            'message': _('Case saved successfully') if created else _('Case already saved'),
            'created': created
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_POST
def api_export_case(request):
    """API endpoint to export cases"""
    try:
        data = json.loads(request.body)
        case_id = data.get('case_id')
        format_type = data.get('format', 'pdf')

        case = get_object_or_404(Case, id=case_id)

        # Mock export implementation
        filename = f"{case.citation.replace(' ', '_')}.{format_type}"
        content = f"Exported case: {case.title}"

        return JsonResponse({
            'success': True,
            'filename': filename,
            'content': content,
            'content_type': 'application/octet-stream'
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def api_process_upload(request):
    """API endpoint to process file uploads"""
    return process_upload(request)


# Helper Functions
def perform_mock_search(query, filters):
    """Mock search implementation"""
    # Generate mock search results
    results = []
    for i in range(random.randint(5, 20)):
        result = {
            'id': uuid.uuid4(),
            'title': f"Legal Case {i+1} - {query.title()}",
            'summary': f"This is a mock case summary related to {query}. The case discusses important legal precedents and judgments.",
            'citation': f"2023 SCC {random.randint(100, 999)}",
            'court': f"High Court of {random.choice(['Delhi', 'Mumbai', 'Chennai', 'Kolkata'])}",
            'judgment_date': datetime.now() - timedelta(days=random.randint(1, 365)),
            'tags': ['Contract Law', 'Commercial Dispute', 'Precedent'],
            'relevance_score': random.randint(60, 95),
        }
        results.append(result)

    # Sort by relevance score
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return results


def generate_mock_suggestions(query):
    """Generate mock search suggestions"""
    suggestions = [
        {
            'text': f"{query} in contract law",
            'type': 'Legal precedent',
            'count': random.randint(10, 100)
        },
        {
            'text': f"{query} commercial dispute",
            'type': 'Case type',
            'count': random.randint(5, 50)
        },
        {
            'text': f"Recent {query} judgments",
            'type': 'Time filter',
            'count': random.randint(20, 80)
        },
    ]
    return suggestions


def generate_mock_analytics(user, period='month'):
    """Generate mock analytics data"""
    return {
        'kpis': [
            {'label': _('Total Searches'), 'value': random.randint(100, 500), 'trend': 'up', 'change': 12},
            {'label': _('Cases Saved'), 'value': random.randint(20, 100), 'trend': 'up', 'change': 8},
            {'label': _('Notes Created'), 'value': random.randint(30, 150), 'trend': 'down', 'change': 3},
            {'label': _('Active Hours'), 'value': f"{random.randint(20, 80)}h", 'trend': 'up', 'change': 15},
        ],
        'trends': [
            {
                'title': _('Search Activity'),
                'data': [
                    {'label': 'Mon', 'value': random.randint(10, 50)},
                    {'label': 'Tue', 'value': random.randint(10, 50)},
                    {'label': 'Wed', 'value': random.randint(10, 50)},
                    {'label': 'Thu', 'value': random.randint(10, 50)},
                    {'label': 'Fri', 'value': random.randint(10, 50)},
                    {'label': 'Sat', 'value': random.randint(5, 25)},
                    {'label': 'Sun', 'value': random.randint(5, 25)},
                ]
            },
            {
                'title': _('Case Categories'),
                'data': [
                    {'label': 'Contract', 'value': random.randint(20, 100)},
                    {'label': 'Property', 'value': random.randint(10, 50)},
                    {'label': 'Family', 'value': random.randint(5, 30)},
                    {'label': 'Criminal', 'value': random.randint(15, 60)},
                ]
            }
        ],
        'predictions': {
            'summary': _('Based on your search patterns, we predict increased interest in contract law cases over the next quarter.'),
            'items': [
                {'title': _('Contract Law Trend'), 'confidence': 85},
                {'title': _('Commercial Disputes'), 'confidence': 72},
                {'title': _('Property Law'), 'confidence': 68},
            ]
        }
    }


def process_uploaded_file(uploaded_file, user):
    """Process uploaded file (mock implementation)"""
    # Mock file processing
    filename = uploaded_file.name
    file_size = uploaded_file.size

    # Simulate processing time
    import time
    time.sleep(1)

    return {
        'success': True,
        'message': _('File {} processed successfully').format(filename),
        'filename': filename,
        'size': file_size,
        'records_processed': random.randint(10, 100)
    }