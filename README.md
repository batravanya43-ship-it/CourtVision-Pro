# CourtVision Pro - AI-Driven Research Engine for Commercial Courts

A comprehensive Django-based web application designed to assist judicial officers in legal research with AI-powered case analysis, predictive insights, and advanced search capabilities.

## 🚀 Features

### Core Functionality
- **Advanced Case Search**: AI-powered search through thousands of legal cases with intelligent suggestions
- **Case Detail Views**: Comprehensive case information with AI-generated summaries
- **Analytics Dashboard**: Predictive analytics and research trend analysis
- **Customization Panel**: Personalized research preferences per suit/case
- **Data Import/Export**: Bulk upload and export capabilities
- **Multilingual Support**: English, Hindi, Tamil, and Telugu language support

### Technical Features
- **Responsive Design**: Mobile-first design using Bootstrap 5
- **AJAX Interactions**: Real-time search suggestions and dynamic content
- **Security**: Role-based access control and data protection
- **Scalability**: Optimized database queries and caching
- **Accessibility**: WCAG 2.1 AA compliant interface

## 📋 System Requirements

- Python 3.8+
- Django 4.2+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- 2GB RAM minimum
- 1GB disk space

## 🛠️ Installation

### Prerequisites
```bash
# Ensure Python 3.8+ is installed
python --version

# Install pip if not available
python -m ensurepip --upgrade
```

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CourtVision-Pro
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On Unix/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up database**
   ```bash
   # Apply migrations
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create demo data**
   ```bash
   python manage.py setup_demo
   ```

7. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

8. **Run development server**
   ```bash
   python manage.py runserver
   ```

9. **Access the application**
   - Open your browser and navigate to `http://localhost:8000`
   - Login with demo credentials: `admin` / `admin123`

## 📁 Project Structure

```
CourtVision-Pro/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── courtvision/             # Django project directory
│   ├── __init__.py
│   ├── settings.py          # Django settings
│   ├── urls.py              # Main URL configuration
│   ├── wsgi.py              # WSGI configuration
│   └── management/          # Custom management commands
│       └── commands/
│           └── setup_demo.py
├── legal_research/          # Main Django app
│   ├── models.py            # Database models
│   ├── views.py             # View functions
│   ├── urls.py              # App URL configuration
│   ├── admin.py             # Django admin configuration
│   └── apps.py              # App configuration
├── templates/               # HTML templates
│   ├── base.html           # Base template
│   └── legal_research/     # App templates
│       ├── landing.html
│       ├── dashboard.html
│       ├── search/
│       ├── cases/
│       └── ...
├── static/                  # Static files
│   ├── css/
│   │   ├── main.css
│   │   ├── dashboard.css
│   │   └── search.css
│   ├── js/
│   │   └── main.js
│   └── images/
└── media/                   # User uploaded files
```

## 🎯 Usage Guide

### Login and Authentication
1. Navigate to the application URL
2. Enter demo credentials (`admin` / `admin123`) or create a new account
3. Select your preferred language from the dropdown

### Dashboard
- **Overview**: View search statistics and recent activity
- **Quick Actions**: Access search, analytics, and customization
- **System Status**: Monitor platform availability

### Advanced Search
1. **Basic Search**: Enter keywords in the search bar
2. **Filters**: Apply court, date range, and case type filters
3. **Suggestions**: Use AI-powered search suggestions
4. **Results**: View ranked results with relevance scores

### Case Analysis
- **View Cases**: Access full case text and metadata
- **AI Summaries**: Review AI-generated case summaries
- **Related Cases**: Find legally similar cases
- **Note Taking**: Add personal notes to cases

### Customization
1. **Suit Selection**: Choose active suits/cases
2. **Preferences**: Set jurisdiction and language preferences
3. **Analysis Focus**: Configure precedent vs. statute weights
4. **Auto-Save**: Preferences are automatically saved

### Analytics
- **KPIs**: Track search activity and usage patterns
- **Trends**: Visualize research trends over time
- **Predictions**: View AI-generated insights

## 🔧 Configuration

### Environment Variables
Key configuration options in `.env`:

```bash
# Django Settings
SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=yourdomain.com

# Database (PostgreSQL recommended for production)
DB_NAME=courtvision
DB_USER=courtvision
DB_PASSWORD=secure_password
DB_HOST=localhost
DB_PORT=5432

# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=app-password
```

### Database Setup

#### SQLite (Development)
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

#### PostgreSQL (Production)
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST'),
        'PORT': config('DB_PORT'),
    }
}
```

## 🔒 Security Features

- **Authentication**: Django's built-in user authentication
- **Authorization**: Role-based access control
- **CSRF Protection**: All forms protected against CSRF attacks
- **SQL Injection Prevention**: Django ORM parameterized queries
- **XSS Protection**: Template auto-escaping
- **Data Encryption**: Sensitive data encryption at rest

## 🌍 Multilingual Support

Supported languages:
- English (en)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)

### Adding New Languages
1. Add language to `settings.py`:
   ```python
   LANGUAGES = [
       ('en', 'English'),
       ('hi', 'हिन्दी (Hindi)'),
       # Add new language here
   ]
   ```

2. Create translation files:
   ```bash
   python manage.py makemessages -l <language_code>
   python manage.py compilemessages
   ```

## 📊 API Endpoints

### Search APIs
- `POST /api/search/suggestions/` - Get search suggestions
- `POST /api/search/results/` - Get paginated search results
- `GET /api/search/filters/` - Get available filters

### User Preference APIs
- `POST /api/preferences/update/` - Update user preferences
- `GET /api/preferences/load/` - Load user preferences

### Case Management APIs
- `POST /api/cases/save/` - Save case to library
- `POST /api/cases/export/` - Export case data
- `POST /api/notes/save/` - Save case notes

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test legal_research

# Run with coverage
pip install coverage
coverage run --source='.' manage.py test
coverage report
```

### Test Coverage
- Model tests: Database operations and validations
- View tests: HTTP responses and logic
- Integration tests: End-to-end workflows
- API tests: AJAX endpoints and JSON responses

## 🚀 Deployment

### Production Deployment

#### Using Gunicorn and Nginx
```bash
# Install Gunicorn
pip install gunicorn

# Run Gunicorn
gunicorn courtvision.wsgi:application

# Systemd service example
[Unit]
Description=CourtVision Pro Django Application
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/CourtVision-Pro
ExecStart=/path/to/venv/bin/gunicorn courtvision.wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["gunicorn", "courtvision.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### Environment Configuration
- **Development**: SQLite database, debug mode enabled
- **Staging**: PostgreSQL, debug disabled, basic monitoring
- **Production**: PostgreSQL, Redis caching, SSL/TLS, monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python manage.py test`
5. Commit changes: `git commit -m 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions small and focused

## 📝 License

This project is licensed under MIT License - see the LICENSE file for details.

## 🆘 Support

### Documentation
- [User Manual](docs/user-manual.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

### Common Issues

**Issue**: Django not found
```bash
# Solution: Activate virtual environment
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows
```

**Issue**: Database migration errors
```bash
# Solution: Reset migrations
python manage.py makemigrations --noinput
python manage.py migrate --fake-initial
```

**Issue**: Static files not loading
```bash
# Solution: Collect static files
python manage.py collectstatic --noinput
```

### Contact
- Email: support@courtvision.com
- Issues: [GitHub Issues](https://github.com/your-org/courtvision-pro/issues)
- Documentation: [Wiki](https://github.com/your-org/courtvision-pro/wiki)

## 🗺️ Roadmap

### Version 1.1 (Planned)
- [ ] Enhanced AI accuracy with machine learning
- [ ] Advanced analytics with predictive modeling
- [ ] Mobile application (iOS/Android)
- [ ] Integration with existing court systems

### Version 1.2 (Future)
- [ ] Voice search capabilities
- [ ] Real-time collaboration features
- [ ] Advanced legal reasoning AI
- [ ] Multi-jurisdictional expansion

## 📈 Analytics and Monitoring

### Performance Metrics
- Page load time < 3 seconds
- Search response time < 2 seconds
- 99.9% uptime availability
- Mobile compatibility score > 95%

### Monitoring Tools
- Django Debug Toolbar (development)
- Sentry error tracking
- Prometheus metrics
- Grafana dashboards

---

**CourtVision Pro** - Empowering judicial officers with AI-powered legal research while maintaining the highest standards of judicial independence and integrity.