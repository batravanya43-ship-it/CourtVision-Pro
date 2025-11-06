from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from .models import (
    HighCourt, UserProfile, Suit, Tag, Case, SearchHistory,
    Customization, UserNote, SavedCase, AnalyticsData
)


@admin.register(HighCourt)
class HighCourtAdmin(admin.ModelAdmin):
    list_display = ['name', 'jurisdiction', 'code', 'established_date', 'is_active']
    list_filter = ['is_active', 'established_date']
    search_fields = ['name', 'jurisdiction', 'code']
    ordering = ['name']


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'
    fk_name = 'user'


class CustomUserAdmin(UserAdmin):
    inlines = (UserProfileInline,)
    list_display = UserAdmin.list_display + ('get_designation', 'get_high_court')

    def get_designation(self, obj):
        try:
            return obj.userprofile.designation
        except UserProfile.DoesNotExist:
            return 'N/A'
    get_designation.short_description = 'Designation'

    def get_high_court(self, obj):
        try:
            return obj.userprofile.high_court.name if obj.userprofile.high_court else 'N/A'
        except UserProfile.DoesNotExist:
            return 'N/A'
    get_high_court.short_description = 'High Court'


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'designation', 'high_court', 'employee_id', 'default_language']
    list_filter = ['high_court', 'default_language', 'designation']
    search_fields = ['user__username', 'user__first_name', 'user__last_name', 'employee_id']
    raw_id_fields = ['user']


@admin.register(Suit)
class SuitAdmin(admin.ModelAdmin):
    list_display = ['name', 'suit_type', 'priority_level', 'created_by', 'is_active', 'created_at']
    list_filter = ['suit_type', 'priority_level', 'is_active', 'created_at']
    search_fields = ['name', 'description']
    raw_id_fields = ['created_by']
    filter_horizontal = ['assigned_users']
    date_hierarchy = 'created_at'


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ['name', 'color_display', 'description']
    search_fields = ['name', 'description']
    list_editable = ['color']

    def color_display(self, obj):
        return format_html(
            '<span style="background-color: {}; padding: 4px 8px; border-radius: 3px; color: white;">{}</span>',
            obj.color, obj.color
        )
    color_display.short_description = 'Color'


@admin.register(Case)
class CaseAdmin(admin.ModelAdmin):
    list_display = ['title', 'citation', 'court', 'judgment_date', 'case_type', 'relevance_score', 'is_published']
    list_filter = ['court', 'case_type', 'judgment_date', 'is_published', 'created_at']
    search_fields = ['title', 'citation', 'petitioners', 'respondents']
    raw_id_fields = ['court']
    filter_horizontal = ['tags']
    date_hierarchy = 'judgment_date'
    readonly_fields = ['id', 'view_count', 'created_at', 'updated_at']

    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'citation', 'court', 'bench', 'case_type')
        }),
        ('Dates', {
            'fields': ('judgment_date', 'decision_date')
        }),
        ('Parties', {
            'fields': ('petitioners', 'respondents')
        }),
        ('Content', {
            'fields': ('case_text', 'headnotes')
        }),
        ('AI Content', {
            'fields': ('ai_summary', 'extracted_principles', 'statutes_cited', 'precedents_cited'),
            'classes': ('collapse',)
        }),
        ('Classification', {
            'fields': ('tags', 'relevance_score', 'is_published')
        }),
        ('Metadata', {
            'fields': ('id', 'view_count', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'query_preview', 'results_count', 'search_time', 'timestamp']
    list_filter = ['timestamp', 'results_count']
    search_fields = ['user__username', 'query_text']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'

    def query_preview(self, obj):
        return obj.query_text[:100] + '...' if len(obj.query_text) > 100 else obj.query_text
    query_preview.short_description = 'Search Query'


@admin.register(Customization)
class CustomizationAdmin(admin.ModelAdmin):
    list_display = ['user', 'suit', 'time_period_focus', 'precedent_statute_weight', 'updated_at']
    list_filter = ['time_period_focus', 'updated_at']
    search_fields = ['user__username', 'suit__name']
    raw_id_fields = ['user', 'suit']
    date_hierarchy = 'updated_at'


@admin.register(UserNote)
class UserNoteAdmin(admin.ModelAdmin):
    list_display = ['user', 'case_preview', 'is_private', 'is_starred', 'updated_at']
    list_filter = ['is_private', 'is_starred', 'updated_at']
    search_fields = ['user__username', 'case__title', 'note_text']
    raw_id_fields = ['user', 'case']
    date_hierarchy = 'updated_at'

    def case_preview(self, obj):
        return obj.case.title[:100] + '...' if len(obj.case.title) > 100 else obj.case.title
    case_preview.short_description = 'Case'


@admin.register(SavedCase)
class SavedCaseAdmin(admin.ModelAdmin):
    list_display = ['user', 'case_preview', 'folder', 'saved_at']
    list_filter = ['folder', 'saved_at']
    search_fields = ['user__username', 'case__title', 'tags']
    raw_id_fields = ['user', 'case']
    date_hierarchy = 'saved_at'

    def case_preview(self, obj):
        return obj.case.title[:100] + '...' if len(obj.case.title) > 100 else obj.case.title
    case_preview.short_description = 'Case'


@admin.register(AnalyticsData)
class AnalyticsDataAdmin(admin.ModelAdmin):
    list_display = ['user', 'analytics_type', 'period_start', 'period_end', 'created_at']
    list_filter = ['analytics_type', 'period_start', 'period_end']
    search_fields = ['user__username']
    raw_id_fields = ['user']
    date_hierarchy = 'period_end'


# Replace the default User admin with our custom one
from django.contrib.auth.models import User
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)