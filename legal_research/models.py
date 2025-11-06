from django.db import models
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _
from django.urls import reverse
import uuid
import json


class HighCourt(models.Model):
    """High Courts in India"""
    name = models.CharField(max_length=200, verbose_name=_("Court Name"))
    jurisdiction = models.CharField(max_length=200, verbose_name=_("Jurisdiction"))
    code = models.CharField(max_length=10, unique=True, verbose_name=_("Court Code"))
    established_date = models.DateField(verbose_name=_("Established Date"))
    is_active = models.BooleanField(default=True, verbose_name=_("Active"))

    class Meta:
        verbose_name = _("High Court")
        verbose_name_plural = _("High Courts")
        ordering = ['name']

    def __str__(self):
        return self.name


class UserProfile(models.Model):
    """Extended user profile for judicial officers"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, verbose_name=_("User"))
    high_court = models.ForeignKey(HighCourt, on_delete=models.SET_NULL, null=True, blank=True, verbose_name=_("High Court"))
    designation = models.CharField(max_length=100, verbose_name=_("Designation"))
    employee_id = models.CharField(max_length=50, unique=True, verbose_name=_("Employee ID"))
    phone_number = models.CharField(max_length=20, blank=True, verbose_name=_("Phone Number"))
    default_language = models.CharField(max_length=10, choices=[
        ('en', 'English'),
        ('hi', 'Hindi'),
        ('ta', 'Tamil'),
        ('te', 'Telugu'),
    ], default='en', verbose_name=_("Default Language"))

    # Notification preferences as JSON
    notification_settings = models.JSONField(default=dict, verbose_name=_("Notification Settings"))

    # User preferences as JSON
    preferences = models.JSONField(default=dict, verbose_name=_("User Preferences"))

    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Created At"))
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Updated At"))

    class Meta:
        verbose_name = _("User Profile")
        verbose_name_plural = _("User Profiles")

    def __str__(self):
        return f"{self.user.get_full_name()} - {self.designation}"

    def get_active_suits(self):
        """Get active suits assigned to this user"""
        return self.suit_assignments.filter(is_active=True)


class Suit(models.Model):
    """Legal suit/case type"""
    name = models.CharField(max_length=200, verbose_name=_("Suit Name"))
    description = models.TextField(verbose_name=_("Description"))
    suit_type = models.CharField(max_length=100, choices=[
        ('civil', 'Civil Suit'),
        ('criminal', 'Criminal Case'),
        ('commercial', 'Commercial Dispute'),
        ('tax', 'Tax Matter'),
        ('constitutional', 'Constitutional Matter'),
        ('other', 'Other'),
    ], verbose_name=_("Suit Type"))

    priority_level = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ], default='medium', verbose_name=_("Priority Level"))

    created_by = models.ForeignKey(UserProfile, on_delete=models.CASCADE, verbose_name=_("Created By"))
    assigned_users = models.ManyToManyField(UserProfile, related_name='suit_assignments', verbose_name=_("Assigned Users"))

    is_active = models.BooleanField(default=True, verbose_name=_("Active"))
    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Created At"))
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Updated At"))

    class Meta:
        verbose_name = _("Suit")
        verbose_name_plural = _("Suits")
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class Tag(models.Model):
    """Tags for categorizing cases"""
    name = models.CharField(max_length=100, unique=True, verbose_name=_("Tag Name"))
    description = models.TextField(blank=True, verbose_name=_("Description"))
    color = models.CharField(max_length=7, default='#007bff', verbose_name=_("Color"))

    class Meta:
        verbose_name = _("Tag")
        verbose_name_plural = _("Tags")
        ordering = ['name']

    def __str__(self):
        return self.name


class Case(models.Model):
    """Legal case/judgment"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Basic case information
    title = models.CharField(max_length=500, verbose_name=_("Case Title"))
    citation = models.CharField(max_length=200, verbose_name=_("Citation"))
    court = models.ForeignKey(HighCourt, on_delete=models.CASCADE, verbose_name=_("Court"))
    bench = models.CharField(max_length=200, blank=True, verbose_name=_("Bench"))

    # Date information
    judgment_date = models.DateField(verbose_name=_("Judgment Date"))
    decision_date = models.DateField(verbose_name=_("Decision Date"))

    # Parties involved
    petitioners = models.TextField(verbose_name=_("Petitioners"))
    respondents = models.TextField(verbose_name=_("Respondents"))

    # Case content
    case_text = models.TextField(verbose_name=_("Full Case Text"))
    headnotes = models.TextField(blank=True, verbose_name=_("Headnotes"))

    # AI-generated content (stored as JSON)
    ai_summary = models.JSONField(default=dict, verbose_name=_("AI Summary"))
    extracted_principles = models.JSONField(default=list, verbose_name=_("Extracted Legal Principles"))
    statutes_cited = models.JSONField(default=list, verbose_name=_("Statutes Cited"))
    precedents_cited = models.JSONField(default=list, verbose_name=_("Precedents Cited"))

    # Classification
    tags = models.ManyToManyField(Tag, blank=True, verbose_name=_("Tags"))
    case_type = models.CharField(max_length=100, choices=[
        ('judgment', 'Judgment'),
        ('order', 'Order'),
        ('interim', 'Interim Order'),
        ('appeal', 'Appeal'),
        ('revision', 'Revision'),
    ], default='judgment', verbose_name=_("Case Type"))

    # Metadata
    relevance_score = models.FloatField(default=0.0, verbose_name=_("Relevance Score"))
    is_published = models.BooleanField(default=False, verbose_name=_("Published"))
    view_count = models.PositiveIntegerField(default=0, verbose_name=_("View Count"))

    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Created At"))
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Updated At"))

    class Meta:
        verbose_name = _("Case")
        verbose_name_plural = _("Cases")
        ordering = ['-judgment_date']
        indexes = [
            models.Index(fields=['court', 'judgment_date']),
            models.Index(fields=['case_type']),
            models.Index(fields=['relevance_score']),
        ]

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('case_detail', kwargs={'case_id': self.id})

    def get_related_cases(self, limit=5):
        """Get related cases based on tags and court"""
        similar_tags = self.tags.all()
        related_cases = Case.objects.filter(
            tags__in=similar_tags
        ).exclude(id=self.id).distinct()[:limit]
        return related_cases

    def generate_summary(self):
        """Generate AI summary (placeholder for actual AI integration)"""
        # This would integrate with an AI service in production
        return {
            'summary': 'AI-generated summary would appear here...',
            'key_points': ['Key point 1', 'Key point 2', 'Key point 3'],
            'decision': 'Brief overview of the decision',
            'implications': 'Legal implications of this case'
        }


class SearchHistory(models.Model):
    """User search history"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name=_("User"))
    query_text = models.TextField(verbose_name=_("Search Query"))

    # Search filters (stored as JSON)
    filters = models.JSONField(default=dict, verbose_name=_("Search Filters"))

    # Search results
    results_count = models.PositiveIntegerField(default=0, verbose_name=_("Results Count"))
    search_time = models.FloatField(default=0.0, verbose_name=_("Search Time (seconds)"))

    timestamp = models.DateTimeField(auto_now_add=True, verbose_name=_("Search Timestamp"))

    class Meta:
        verbose_name = _("Search History")
        verbose_name_plural = _("Search Histories")
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.query_text[:50]}"

    def get_similar_searches(self, limit=10):
        """Get similar searches based on query text"""
        from django.db.models import Q
        return SearchHistory.objects.filter(
            Q(query_text__icontains=self.query_text[:20]) |
            Q(filters=self.filters)
        ).exclude(id=self.id)[:limit]


class Customization(models.Model):
    """User customization preferences per suit"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name=_("User"))
    suit = models.ForeignKey(Suit, on_delete=models.CASCADE, verbose_name=_("Suit"))

    # Jurisdiction emphasis (stored as JSON)
    jurisdiction_emphasis = models.JSONField(default=dict, verbose_name=_("Jurisdiction Emphasis"))

    # Language preferences
    language_preferences = models.JSONField(default=list, verbose_name=_("Language Preferences"))

    # Analysis weights
    precedent_statute_weight = models.FloatField(default=0.5, verbose_name=_("Precedent vs Statute Weight"))
    time_period_focus = models.CharField(max_length=50, choices=[
        ('recent', 'Recent Cases (Last 5 years)'),
        ('medium', 'Medium Period (Last 10 years)'),
        ('historical', 'Historical (All Time)'),
        ('custom', 'Custom Range'),
    ], default='recent', verbose_name=_("Time Period Focus"))

    # Analysis focus areas
    analysis_focus_areas = models.JSONField(default=list, verbose_name=_("Analysis Focus Areas"))

    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Created At"))
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Updated At"))

    class Meta:
        verbose_name = _("Customization")
        verbose_name_plural = _("Customizations")
        unique_together = ['user', 'suit']
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.user.username} - {self.suit.name}"


class UserNote(models.Model):
    """User notes on cases"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name=_("User"))
    case = models.ForeignKey(Case, on_delete=models.CASCADE, verbose_name=_("Case"))
    note_text = models.TextField(verbose_name=_("Note Text"))

    # Note metadata
    is_private = models.BooleanField(default=True, verbose_name=_("Private"))
    is_starred = models.BooleanField(default=False, verbose_name=_("Starred"))

    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Created At"))
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Updated At"))

    class Meta:
        verbose_name = _("User Note")
        verbose_name_plural = _("User Notes")
        ordering = ['-updated_at']
        unique_together = ['user', 'case']

    def __str__(self):
        return f"{self.user.username} - {self.case.title[:50]}"


class SavedCase(models.Model):
    """Cases saved by users"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name=_("User"))
    case = models.ForeignKey(Case, on_delete=models.CASCADE, verbose_name=_("Case"))

    # Categorization
    folder = models.CharField(max_length=100, default='General', verbose_name=_("Folder"))
    tags = models.CharField(max_length=500, blank=True, verbose_name=_("User Tags"))

    saved_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Saved At"))

    class Meta:
        verbose_name = _("Saved Case")
        verbose_name_plural = _("Saved Cases")
        unique_together = ['user', 'case']
        ordering = ['-saved_at']

    def __str__(self):
        return f"{self.user.username} - {self.case.title[:50]}"


class AnalyticsData(models.Model):
    """Analytics and statistics data"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name=_("User"))

    # Analytics type and data
    analytics_type = models.CharField(max_length=50, choices=[
        ('search_trends', 'Search Trends'),
        ('case_views', 'Case Views'),
        ('user_activity', 'User Activity'),
        ('system_usage', 'System Usage'),
        ('predictions', 'Predictions'),
    ], verbose_name=_("Analytics Type"))

    # Data stored as JSON
    data = models.JSONField(verbose_name=_("Analytics Data"))

    # Time period
    period_start = models.DateField(verbose_name=_("Period Start"))
    period_end = models.DateField(verbose_name=_("Period End"))

    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Created At"))

    class Meta:
        verbose_name = _("Analytics Data")
        verbose_name_plural = _("Analytics Data")
        ordering = ['-period_end']
        indexes = [
            models.Index(fields=['user', 'analytics_type', 'period_end']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.analytics_type} ({self.period_start} to {self.period_end})"