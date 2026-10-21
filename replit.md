# IVF Journey Tracker

## Overview

The IVF Journey Tracker is a comprehensive web application designed to help patients and doctors monitor, track, and manage In Vitro Fertilization (IVF) treatments. The application leverages AI-powered features to provide personalized predictions, wellness guidance, and patient support throughout the fertility journey. Built with Flask and modern web technologies, it offers separate dashboards for patients and healthcare providers, along with intelligent chatbot assistance and predictive analytics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Web Framework Architecture
The application follows a traditional Model-View-Controller (MVC) pattern using Flask as the web framework. This choice provides flexibility and rapid development capabilities while maintaining clean separation of concerns. The main application is structured with `app.py` handling configuration and initialization, `main.py` serving as the entry point, and `routes.py` managing all HTTP endpoints and business logic.

### Database Architecture
The system uses SQLAlchemy ORM with SQLite as the default database, configurable to PostgreSQL via environment variables. The database schema is designed around core entities: Users (supporting both patients and doctors), PatientData for medical information, IVFCycle for treatment tracking, WellnessLog for daily health monitoring, and supporting tables for medication reminders and chat messages. This relational structure ensures data integrity while allowing for complex queries across treatment cycles.

### Authentication and Authorization
User authentication is implemented using session-based login with password hashing via Werkzeug's security utilities. The system supports role-based access control distinguishing between patient and doctor user types, with appropriate route protection and dashboard differentiation. Session management handles user state across requests with configurable session secrets.

### AI Integration Architecture
The application integrates multiple AI capabilities through OpenAI's API services. The AI chatbot provides 24/7 patient support with context-aware responses based on user profiles and medical history. Additional AI features include medical image generation, personalized nutrition planning, and yoga routine recommendations. All AI interactions are handled through a dedicated service layer that manages API calls and response processing.

### Prediction Engine
A rule-based prediction system calculates IVF success rates using multiple patient factors including age, BMI, medical history, and previous treatment outcomes. The algorithm provides confidence scores and contributing factor analysis, offering patients data-driven insights into their treatment prospects while maintaining appropriate medical disclaimers.

### Frontend Architecture
The frontend uses a responsive design built with Bootstrap 5, providing mobile-first user interfaces. JavaScript handles interactive features including form validation, chart rendering with Chart.js, and real-time chatbot functionality. The template system uses Jinja2 with a base template structure ensuring consistent styling and navigation across all pages.

### File Upload System
Medical document management is handled through a secure file upload system supporting multiple file formats (PDF, images, documents). Files are stored in a dedicated uploads directory with size limitations and file type validation for security. The system integrates with the database to track document metadata and user associations.

## External Dependencies

### Core Framework Dependencies
- **Flask** - Primary web framework providing routing, templating, and request handling
- **Flask-SQLAlchemy** - Database ORM for data modeling and query management
- **Werkzeug** - WSGI utilities including password hashing and file upload handling

### AI and Machine Learning Services
- **OpenAI API** - Powers the intelligent chatbot, medical image generation, and personalized content creation
- **GPT-5 Model** - Latest language model for natural language processing and medical conversation handling

### Frontend Technologies
- **Bootstrap 5** - CSS framework for responsive design and component styling
- **Chart.js** - JavaScript library for data visualization and progress tracking
- **Font Awesome** - Icon library for consistent UI elements

### Database Systems
- **SQLite** - Default embedded database for development and small deployments
- **PostgreSQL** - Production-ready database option configured via environment variables

### Security and Session Management
- **Flask Session** - Server-side session handling for user authentication state
- **ProxyFix** - Middleware for handling reverse proxy headers in production deployments

### Development and Deployment Tools
- **Python 3.x** - Runtime environment with standard libraries for date/time handling and logging
- **Environment Variables** - Configuration management for database connections and API keys