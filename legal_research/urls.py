from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'legal_research'

urlpatterns = [
    # Landing page (root)
    path('', views.landing_page, name='landing'),

    # Authentication
    path('login/', auth_views.LoginView.as_view(template_name='legal_research/auth/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),

    # Main application views
    path('dashboard/', views.dashboard, name='dashboard'),

    # Search functionality
    path('search/', views.advanced_search, name='search'),
    path('search/results/', views.search_results, name='search_results'),
    path('search/ajax/suggestions/', views.ajax_search_suggestions, name='ajax_search_suggestions'),
    path('search/ajax/results/', views.ajax_search_results, name='ajax_search_results'),

    # Case detail views
    path('case/<uuid:case_id>/', views.case_detail, name='case_detail'),
    path('case/<uuid:case_id>/save/', views.save_case, name='save_case'),
    path('case/<uuid:case_id>/export/', views.export_case, name='export_case'),
    path('case/<uuid:case_id>/note/', views.save_case_note, name='save_case_note'),

    # Customization
    path('customization/', views.customization_panel, name='customization'),
    path('customization/update/', views.update_customization, name='update_customization'),
    path('customization/load/', views.load_customization, name='load_customization'),

    # Analytics
    path('analytics/', views.analytics_dashboard, name='analytics'),
    path('analytics/data/', views.analytics_data, name='analytics_data'),

    # Data upload
    path('upload/', views.data_upload, name='upload'),
    path('upload/process/', views.process_upload, name='process_upload'),

    # User profile and settings
    path('profile/', views.user_profile, name='profile'),
    path('profile/update/', views.update_profile, name='update_profile'),

    # Help and ethical guidelines
    path('help/', views.help_page, name='help'),

    # API endpoints
    path('api/preferences/update/', views.api_update_preferences, name='api_update_preferences'),
    path('api/preferences/load/', views.api_load_preferences, name='api_load_preferences'),
    path('api/notes/save/', views.api_save_note, name='api_save_note'),
    path('api/cases/save/', views.api_save_case, name='api_save_case'),
    path('api/cases/export/', views.api_export_case, name='api_export_case'),
    path('api/upload/process/', views.api_process_upload, name='api_process_upload'),
]