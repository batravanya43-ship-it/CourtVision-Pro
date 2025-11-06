// CourtVision Pro - Main JavaScript File

// ===== Global Variables =====
window.CourtVision = {
    csrfToken: window.CSRF_TOKEN,
    baseUrl: window.location.origin,
    currentLanguage: document.documentElement.lang,
    isLoading: false
};

// ===== Utility Functions =====
const Utils = {
    // Show loading spinner
    showLoading() {
        CourtVision.isLoading = true;
        const spinner = document.getElementById('loadingSpinner');
        if (spinner) {
            spinner.style.display = 'block';
        }
        document.body.classList.add('loading');
    },

    // Hide loading spinner
    hideLoading() {
        CourtVision.isLoading = false;
        const spinner = document.getElementById('loadingSpinner');
        if (spinner) {
            spinner.style.display = 'none';
        }
        document.body.classList.remove('loading');
    },

    // Make AJAX request
    async request(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': CourtVision.csrfToken
            }
        };

        const mergedOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, mergedOptions);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || 'Request failed');
            }

            return data;
        } catch (error) {
            console.error('AJAX Error:', error);
            this.showAlert(error.message, 'danger');
            throw error;
        }
    },

    // Show alert message
    showAlert(message, type = 'info', duration = 5000) {
        const alertContainer = document.querySelector('.container');
        if (!alertContainer) return;

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        alertContainer.insertBefore(alertDiv, alertContainer.firstChild);

        // Auto dismiss after duration
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, duration);
    },

    // Format date
    formatDate(dateString, format = 'short') {
        const date = new Date(dateString);
        const options = {
            short: { year: 'numeric', month: 'short', day: 'numeric' },
            long: { year: 'numeric', month: 'long', day: 'numeric', year: 'numeric' },
            time: { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }
        };

        return date.toLocaleDateString(CourtVision.currentLanguage, options[format] || options.short);
    },

    // Debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Get URL parameters
    getUrlParam(name) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(name);
    },

    // Set URL parameter
    setUrlParam(name, value) {
        const url = new URL(window.location);
        url.searchParams.set(name, value);
        window.history.pushState({}, '', url);
    }
};

// ===== Search Functionality =====
const Search = {
    init() {
        this.setupSearchInput();
        this.setupFilters();
        this.setupSearchForm();
    },

    setupSearchInput() {
        const searchInput = document.getElementById('searchInput');
        const suggestionsContainer = document.getElementById('searchSuggestions');

        if (!searchInput || !suggestionsContainer) return;

        const debouncedSearch = Utils.debounce(async (query) => {
            if (query.length < 2) {
                suggestionsContainer.style.display = 'none';
                return;
            }

            try {
                const response = await Utils.request('/api/search/suggestions/', {
                    method: 'POST',
                    body: JSON.stringify({ query })
                });

                this.displaySuggestions(response.suggestions);
            } catch (error) {
                console.error('Search suggestions error:', error);
            }
        }, 300);

        searchInput.addEventListener('input', (e) => {
            debouncedSearch(e.target.value);
        });

        // Hide suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !suggestionsContainer.contains(e.target)) {
                suggestionsContainer.style.display = 'none';
            }
        });
    },

    displaySuggestions(suggestions) {
        const container = document.getElementById('searchSuggestions');
        if (!container) return;

        if (suggestions.length === 0) {
            container.style.display = 'none';
            return;
        }

        container.innerHTML = suggestions.map((suggestion, index) => `
            <div class="search-suggestion-item" data-index="${index}">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <strong>${this.highlightMatch(suggestion.text, this.getCurrentQuery())}</strong>
                        <div class="small text-muted">${suggestion.type}</div>
                    </div>
                    ${suggestion.count ? `<span class="badge bg-secondary">${suggestion.count}</span>` : ''}
                </div>
            </div>
        `).join('');

        container.style.display = 'block';

        // Add click handlers to suggestions
        container.querySelectorAll('.search-suggestion-item').forEach(item => {
            item.addEventListener('click', () => {
                const searchInput = document.getElementById('searchInput');
                if (searchInput) {
                    searchInput.value = item.textContent.trim();
                    container.style.display = 'none';
                    this.performSearch();
                }
            });
        });
    },

    highlightMatch(text, query) {
        if (!query) return text;
        const regex = new RegExp(`(${query})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    },

    getCurrentQuery() {
        const searchInput = document.getElementById('searchInput');
        return searchInput ? searchInput.value : '';
    },

    setupFilters() {
        const filterCheckboxes = document.querySelectorAll('.search-filter');
        filterCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateActiveFilters();
            });
        });
    },

    updateActiveFilters() {
        const activeFilters = document.querySelectorAll('.search-filter:checked');
        const activeFiltersContainer = document.getElementById('activeFilters');

        if (!activeFiltersContainer) return;

        if (activeFilters.length === 0) {
            activeFiltersContainer.innerHTML = '<span class="text-muted">No filters applied</span>';
            return;
        }

        activeFiltersContainer.innerHTML = Array.from(activeFilters).map(filter => `
            <span class="badge bg-primary me-1">
                ${filter.dataset.label}
                <button type="button" class="btn-close btn-close-white ms-1" data-filter="${filter.name}"></button>
            </span>
        `).join('');

        // Add click handlers to remove filters
        activeFiltersContainer.querySelectorAll('.btn-close').forEach(btn => {
            btn.addEventListener('click', () => {
                const filterName = btn.dataset.filter;
                const checkbox = document.querySelector(`.search-filter[name="${filterName}"]`);
                if (checkbox) {
                    checkbox.checked = false;
                    this.updateActiveFilters();
                }
            });
        });
    },

    setupSearchForm() {
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.performSearch();
            });
        }
    },

    async performSearch() {
        const query = this.getCurrentQuery();
        const filters = this.getSearchFilters();

        if (!query.trim()) {
            Utils.showAlert('Please enter a search query', 'warning');
            return;
        }

        Utils.showLoading();

        try {
            const response = await Utils.request('/api/search/results/', {
                method: 'POST',
                body: JSON.stringify({ query, filters })
            });

            this.displaySearchResults(response);
            this.updateUrl(query, filters);
        } catch (error) {
            console.error('Search error:', error);
        } finally {
            Utils.hideLoading();
        }
    },

    getSearchFilters() {
        const filters = {};
        const filterCheckboxes = document.querySelectorAll('.search-filter:checked');

        filterCheckboxes.forEach(checkbox => {
            filters[checkbox.name] = checkbox.value;
        });

        // Get date range filters
        const dateFrom = document.getElementById('dateFrom');
        const dateTo = document.getElementById('dateTo');
        if (dateFrom && dateFrom.value) filters.date_from = dateFrom.value;
        if (dateTo && dateTo.value) filters.date_to = dateTo.value;

        return filters;
    },

    displaySearchResults(response) {
        const resultsContainer = document.getElementById('searchResults');
        const paginationContainer = document.getElementById('searchPagination');

        if (!resultsContainer) return;

        if (response.results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="text-center py-5">
                    <i class="bi bi-search display-1 text-muted"></i>
                    <h3 class="mt-3">No results found</h3>
                    <p class="text-muted">Try adjusting your search terms or filters</p>
                </div>
            `;
            if (paginationContainer) paginationContainer.innerHTML = '';
            return;
        }

        resultsContainer.innerHTML = response.results.map(result => `
            <div class="search-result-item card mb-3">
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <h5 class="card-title">
                                <a href="/case/${result.id}/" class="text-decoration-none">
                                    ${result.title}
                                </a>
                            </h5>
                            <p class="card-text text-truncate-2">${result.summary}</p>
                            <div class="d-flex flex-wrap gap-1 mb-2">
                                ${result.tags.map(tag => `<span class="tag tag-secondary">${tag}</span>`).join('')}
                            </div>
                            <small class="text-muted">
                                <i class="bi bi-calendar3 me-1"></i>${Utils.formatDate(result.judgment_date)}
                                <span class="ms-3"><i class="bi bi-building me-1"></i>${result.court}</span>
                            </small>
                        </div>
                        <div class="col-md-4 text-end">
                            <div class="relevance-score mb-2">
                                <div class="relevance-score-indicator" style="left: ${result.relevance_score}%"></div>
                            </div>
                            <small class="text-muted">Relevance: ${Math.round(result.relevance_score)}%</small>
                            <div class="mt-2">
                                <button class="btn btn-sm btn-outline-primary me-1" onclick="Search.saveCase('${result.id}')">
                                    <i class="bi bi-bookmark"></i>
                                </button>
                                <button class="btn btn-sm btn-outline-secondary" onclick="Search.exportCase('${result.id}')">
                                    <i class="bi bi-download"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');

        // Display pagination if available
        if (paginationContainer && response.pagination) {
            this.displayPagination(response.pagination, paginationContainer);
        }
    },

    displayPagination(pagination, container) {
        if (pagination.total_pages <= 1) {
            container.innerHTML = '';
            return;
        }

        let paginationHTML = '<nav><ul class="pagination justify-content-center">';

        // Previous button
        if (pagination.has_previous) {
            paginationHTML += `<li class="page-item">
                <a class="page-link" href="#" data-page="${pagination.current_page - 1}">Previous</a>
            </li>`;
        }

        // Page numbers
        for (let i = 1; i <= pagination.total_pages; i++) {
            if (i === pagination.current_page) {
                paginationHTML += `<li class="page-item active">
                    <span class="page-link">${i}</span>
                </li>`;
            } else {
                paginationHTML += `<li class="page-item">
                    <a class="page-link" href="#" data-page="${i}">${i}</a>
                </li>`;
            }
        }

        // Next button
        if (pagination.has_next) {
            paginationHTML += `<li class="page-item">
                <a class="page-link" href="#" data-page="${pagination.current_page + 1}">Next</a>
            </li>`;
        }

        paginationHTML += '</ul></nav>';
        container.innerHTML = paginationHTML;

        // Add click handlers
        container.querySelectorAll('.page-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = e.target.dataset.page;
                this.performSearch(page);
            });
        });
    },

    updateUrl(query, filters) {
        const url = new URL(window.location);
        url.searchParams.set('q', query);

        // Clear existing filters
        Object.keys([...url.searchParams.keys()]).forEach(key => {
            if (key !== 'q') url.searchParams.delete(key);
        });

        // Add new filters
        Object.entries(filters).forEach(([key, value]) => {
            url.searchParams.set(key, value);
        });

        window.history.pushState({}, '', url);
    },

    async saveCase(caseId) {
        try {
            await Utils.request('/api/cases/save/', {
                method: 'POST',
                body: JSON.stringify({ case_id: caseId })
            });
            Utils.showAlert('Case saved successfully', 'success');
        } catch (error) {
            console.error('Save case error:', error);
        }
    },

    async exportCase(caseId) {
        try {
            const response = await Utils.request('/api/cases/export/', {
                method: 'POST',
                body: JSON.stringify({ case_id: caseId })
            });

            // Create download link
            const blob = new Blob([response.content], { type: response.content_type });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = response.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            Utils.showAlert('Case exported successfully', 'success');
        } catch (error) {
            console.error('Export case error:', error);
        }
    }
};

// ===== Case Detail Functionality =====
const CaseDetail = {
    init() {
        this.setupNoteSaving();
        this.setupRelatedCases();
        this.setupCollapsibleSections();
    },

    setupNoteSaving() {
        const noteTextarea = document.getElementById('caseNote');
        const saveNoteBtn = document.getElementById('saveNoteBtn');

        if (!noteTextarea || !saveNoteBtn) return;

        const debouncedSave = Utils.debounce(async () => {
            const noteText = noteTextarea.value.trim();
            const caseId = noteTextarea.dataset.caseId;

            if (!caseId) return;

            try {
                await Utils.request('/api/notes/save/', {
                    method: 'POST',
                    body: JSON.stringify({ case_id: caseId, note_text: noteText })
                });

                saveNoteBtn.innerHTML = '<i class="bi bi-check-circle"></i> Saved';
                saveNoteBtn.classList.remove('btn-primary');
                saveNoteBtn.classList.add('btn-success');

                setTimeout(() => {
                    saveNoteBtn.innerHTML = '<i class="bi bi-save"></i> Save Note';
                    saveNoteBtn.classList.remove('btn-success');
                    saveNoteBtn.classList.add('btn-primary');
                }, 2000);
            } catch (error) {
                console.error('Save note error:', error);
            }
        }, 1000);

        noteTextarea.addEventListener('input', debouncedSave);
        saveNoteBtn.addEventListener('click', debouncedSave);
    },

    setupRelatedCases() {
        const loadRelatedBtn = document.getElementById('loadRelatedCases');
        const relatedCasesContainer = document.getElementById('relatedCasesContainer');

        if (!loadRelatedBtn || !relatedCasesContainer) return;

        loadRelatedBtn.addEventListener('click', async () => {
            const caseId = loadRelatedBtn.dataset.caseId;

            Utils.showLoading();
            try {
                const response = await Utils.request(`/api/cases/${caseId}/related/`);
                this.displayRelatedCases(response.related_cases);
            } catch (error) {
                console.error('Load related cases error:', error);
            } finally {
                Utils.hideLoading();
            }
        });
    },

    displayRelatedCases(cases) {
        const container = document.getElementById('relatedCasesContainer');
        if (!container) return;

        container.innerHTML = cases.map(case_ => `
            <div class="card mb-2">
                <div class="card-body">
                    <h6 class="card-title">
                        <a href="/case/${case_.id}/" class="text-decoration-none">
                            ${case_.title}
                        </a>
                    </h6>
                    <p class="card-text small text-truncate-2">${case_.summary}</p>
                    <small class="text-muted">
                        ${Utils.formatDate(case_.judgment_date)} • ${case_.court}
                    </small>
                </div>
            </div>
        `).join('');
    },

    setupCollapsibleSections() {
        const toggleButtons = document.querySelectorAll('.toggle-section');

        toggleButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetId = button.dataset.target;
                const targetSection = document.getElementById(targetId);
                const icon = button.querySelector('i');

                if (targetSection) {
                    targetSection.classList.toggle('collapse');
                    icon.classList.toggle('bi-chevron-down');
                    icon.classList.toggle('bi-chevron-up');
                }
            });
        });
    }
};

// ===== Customization Panel =====
const Customization = {
    init() {
        this.setupSuitSelector();
        this.setupPreferenceSliders();
        this.setupAutoSave();
    },

    setupSuitSelector() {
        const suitSelect = document.getElementById('suitSelect');
        if (suitSelect) {
            suitSelect.addEventListener('change', () => {
                this.loadSuitPreferences(suitSelect.value);
            });
        }
    },

    setupPreferenceSliders() {
        const sliders = document.querySelectorAll('.preference-slider input[type="range"]');

        sliders.forEach(slider => {
            const valueDisplay = document.getElementById(slider.id + 'Value');

            slider.addEventListener('input', () => {
                if (valueDisplay) {
                    valueDisplay.textContent = slider.value + '%';
                }
            });
        });
    },

    setupAutoSave() {
        const form = document.getElementById('customizationForm');
        if (!form) return;

        const debouncedSave = Utils.debounce(async () => {
            const formData = new FormData(form);
            const preferences = Object.fromEntries(formData.entries());

            // Convert numeric values
            Object.keys(preferences).forEach(key => {
                if (preferences[key].match(/^\d+$/)) {
                    preferences[key] = parseInt(preferences[key]);
                }
            });

            try {
                await Utils.request('/api/preferences/update/', {
                    method: 'POST',
                    body: JSON.stringify(preferences)
                });

                Utils.showAlert('Preferences saved', 'success');
            } catch (error) {
                console.error('Save preferences error:', error);
            }
        }, 2000);

        form.addEventListener('change', debouncedSave);
    },

    async loadSuitPreferences(suitId) {
        if (!suitId) return;

        try {
            const response = await Utils.request(`/api/preferences/load/?suit_id=${suitId}`);
            this.populatePreferencesForm(response.preferences);
        } catch (error) {
            console.error('Load preferences error:', error);
        }
    },

    populatePreferencesForm(preferences) {
        const form = document.getElementById('customizationForm');
        if (!form) return;

        Object.entries(preferences).forEach(([key, value]) => {
            const element = form.querySelector(`[name="${key}"]`);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = Boolean(value);
                } else if (element.type === 'range') {
                    element.value = value;
                    const valueDisplay = document.getElementById(element.id + 'Value');
                    if (valueDisplay) {
                        valueDisplay.textContent = value + '%';
                    }
                } else {
                    element.value = value;
                }
            }
        });
    }
};

// ===== Upload Functionality =====
const Upload = {
    init() {
        this.setupFileUpload();
        this.setupProgressTracking();
    },

    setupFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        if (!uploadArea || !fileInput) return;

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });

        // File selection
        fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });
    },

    handleFiles(files) {
        if (files.length === 0) return;

        const validTypes = ['application/pdf', 'text/plain', 'application/vnd.ms-excel',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           'application/json'];

        Array.from(files).forEach(file => {
            if (!validTypes.includes(file.type)) {
                Utils.showAlert(`Invalid file type: ${file.name}`, 'danger');
                return;
            }

            this.uploadFile(file);
        });
    },

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const progressContainer = this.createProgressElement(file.name);

        try {
            const response = await fetch('/api/upload/process/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': CourtVision.csrfToken
                },
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.updateProgress(progressContainer, 100, 'success', result.message);
            } else {
                this.updateProgress(progressContainer, 0, 'danger', result.message);
            }
        } catch (error) {
            this.updateProgress(progressContainer, 0, 'danger', 'Upload failed');
            console.error('Upload error:', error);
        }
    },

    createProgressElement(filename) {
        const container = document.getElementById('uploadProgress');
        if (!container) return null;

        const progressDiv = document.createElement('div');
        progressDiv.className = 'progress-container mb-3';
        progressDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-1">
                <span class="filename">${filename}</span>
                <span class="status text-muted">Uploading...</span>
            </div>
            <div class="progress">
                <div class="progress-bar" role="progressbar" style="width: 0%"></div>
            </div>
        `;

        container.appendChild(progressDiv);
        return progressDiv;
    },

    updateProgress(container, percentage, status, message) {
        if (!container) return;

        const progressBar = container.querySelector('.progress-bar');
        const statusElement = container.querySelector('.status');

        if (progressBar) {
            progressBar.style.width = percentage + '%';
            progressBar.className = `progress-bar bg-${status}`;
        }

        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status text-${status}`;
        }
    },

    setupProgressTracking() {
        // Simulate progress for demonstration
        // In production, this would be handled by server-sent events or WebSocket
    }
};

// ===== Analytics Charts =====
const Analytics = {
    init() {
        this.setupChartContainers();
        this.loadAnalyticsData();
    },

    setupChartContainers() {
        // Chart containers will be set up based on the specific charting library used
        // For now, we'll create simple CSS-based charts
    },

    async loadAnalyticsData() {
        try {
            const response = await Utils.request('/api/analytics/data/');
            this.displayAnalytics(response.data);
        } catch (error) {
            console.error('Load analytics error:', error);
        }
    },

    displayAnalytics(data) {
        this.displayKPIs(data.kpis);
        this.displayTrends(data.trends);
        this.displayPredictions(data.predictions);
    },

    displayKPIs(kpis) {
        const kpiContainer = document.getElementById('kpiContainer');
        if (!kpiContainer || !kpis) return;

        kpiContainer.innerHTML = kpis.map(kpi => `
            <div class="col-md-3 mb-4">
                <div class="kpi-card">
                    <div class="kpi-number">${kpi.value}</div>
                    <div class="kpi-label">${kpi.label}</div>
                    <div class="mt-2">
                        <small class="text-${kpi.trend === 'up' ? 'success' : 'danger'}">
                            <i class="bi bi-arrow-${kpi.trend}"></i> ${kpi.change}%
                        </small>
                    </div>
                </div>
            </div>
        `).join('');
    },

    displayTrends(trends) {
        // Simple CSS-based trend visualization
        const trendsContainer = document.getElementById('trendsContainer');
        if (!trendsContainer || !trends) return;

        trendsContainer.innerHTML = trends.map(trend => `
            <div class="chart-container">
                <h5>${trend.title}</h5>
                <div class="trend-chart">
                    ${this.createSimpleBarChart(trend.data)}
                </div>
            </div>
        `).join('');
    },

    createSimpleBarChart(data) {
        const maxValue = Math.max(...data.map(d => d.value));

        return `
            <div class="simple-bar-chart">
                ${data.map(item => `
                    <div class="bar-item">
                        <div class="bar-label">${item.label}</div>
                        <div class="bar-container">
                            <div class="bar" style="width: ${(item.value / maxValue) * 100}%"></div>
                        </div>
                        <div class="bar-value">${item.value}</div>
                    </div>
                `).join('')}
            </div>
        `;
    },

    displayPredictions(predictions) {
        const predictionsContainer = document.getElementById('predictionsContainer');
        if (!predictionsContainer || !predictions) return;

        predictionsContainer.innerHTML = `
            <div class="alert alert-info">
                <h6><i class="bi bi-cpu me-2"></i>AI Predictions</h6>
                <p class="mb-0">${predictions.summary}</p>
            </div>
            <div class="row">
                ${predictions.items.map(pred => `
                    <div class="col-md-4 mb-3">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title">${pred.title}</h6>
                                <div class="prediction-confidence">
                                    <small class="text-muted">Confidence:</small>
                                    <div class="progress mt-1">
                                        <div class="progress-bar" style="width: ${pred.confidence}%"></div>
                                    </div>
                                    <small class="text-muted">${pred.confidence}%</small>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
};

// ===== Initialize Application =====
document.addEventListener('DOMContentLoaded', () => {
    // Initialize components based on current page
    const currentPage = document.body.dataset.page;

    switch (currentPage) {
        case 'search':
            Search.init();
            break;
        case 'case_detail':
            CaseDetail.init();
            break;
        case 'customization':
            Customization.init();
            break;
        case 'upload':
            Upload.init();
            break;
        case 'analytics':
            Analytics.init();
            break;
        default:
            // Initialize common functionality
            Search.init();
    }

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    console.log('CourtVision Pro initialized');
});

// Export for use in other files
window.CourtVisionPro = {
    Utils,
    Search,
    CaseDetail,
    Customization,
    Upload,
    Analytics
};