# models.py - FINAL CORRECTED VERSION

from datetime import datetime, date, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer as Serializer
from flask import current_app
from database import db


# --- Core Models ---

class User(db.Model):
    """
    User model for authentication and role management (Patient/Doctor/Admin).
    """
    __tablename__ = 'user' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    first_name = db.Column(db.String(64), nullable=False)
    last_name = db.Column(db.String(64), nullable=False)
    date_of_birth = db.Column(db.Date)
    phone = db.Column(db.String(20))
    
    # 'patient', 'doctor', or 'admin'
    user_type = db.Column(db.String(10), default='patient', nullable=False)
    
    # Link to a clinic if the user is a doctor
    clinic_id = db.Column(db.Integer, db.ForeignKey('clinic.id'), nullable=True) # References 'clinic' table
    
    # Use timezone-aware object for modern stability (FIX)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships (one-to-many/one-to-one)
    patient_data = db.relationship('PatientData', backref='owner', uselist=False, cascade="all, delete-orphan")
    ivf_cycles = db.relationship('IVFCycle', backref='patient', cascade="all, delete-orphan")
    medication_reminders = db.relationship('MedicationReminder', backref='user', cascade="all, delete-orphan")
    wellness_logs = db.relationship('WellnessLog', backref='user', cascade="all, delete-orphan")
    predictions = db.relationship('Prediction', backref='user', cascade="all, delete-orphan")
    chat_messages = db.relationship('ChatMessage', backref='user', cascade="all, delete-orphan")
    medical_documents = db.relationship('MedicalDocument', backref='user', cascade="all, delete-orphan")
    partner_connections_as_patient = db.relationship('PartnerConnection', foreign_keys='PartnerConnection.patient_id', backref='patient_user', cascade="all, delete-orphan")
    partner_connections_as_partner = db.relationship('PartnerConnection', foreign_keys='PartnerConnection.partner_id', backref='partner_user', cascade="all, delete-orphan")
    partner_invitations = db.relationship('PartnerInvitation', backref='patient', cascade="all, delete-orphan")
    
    def set_password(self, password):
        """Hashes the password for secure storage."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Checks the provided password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def get_reset_token(self):
        """Generates a secure, timed token for password reset."""
        s = Serializer(current_app.config['SECRET_KEY'])
        return s.dumps({'user_id': self.id})

    @staticmethod
    def verify_reset_token(token, expires_sec=1800):
        """
        Verifies the password reset token.
        Returns the User object if the token is valid, otherwise None.
        """
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token, max_age=expires_sec)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        return f'<User {self.username} ({self.user_type})>'


class PatientData(db.Model):
    """
    Stores comprehensive medical and lifestyle data for AI analysis.
    One-to-one relationship with User (only users with user_type='patient').
    """
    __tablename__ = 'patient_data' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False) # References 'user' table

    # Physical/Hormonal Data
    age = db.Column(db.Integer)
    height = db.Column(db.Float)  # in cm
    weight = db.Column(db.Float)  # in kg
    bmi = db.Column(db.Float)
    amh_level = db.Column(db.Float) # Anti-Mullerian Hormone (ng/mL)
    fsh_level = db.Column(db.Float) # Follicle-Stimulating Hormone (mIU/mL)
    
    # History
    previous_pregnancies = db.Column(db.Integer, default=0)
    previous_ivf_cycles = db.Column(db.Integer, default=0)
    diagnosis = db.Column(db.Text) # Primary cause of infertility
    medical_history = db.Column(db.Text) # PCOS, Endometriosis, Thyroid issues, etc.
    allergies = db.Column(db.Text)
    
    # Partner & Lifestyle
    partner_age = db.Column(db.Integer)
    partner_diagnosis = db.Column(db.String(255))
    lifestyle_factors = db.Column(db.Text) # Smoking, alcohol, diet, exercise habits

    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def calculate_bmi(self):
        """Calculates BMI (Body Mass Index) from height and weight."""
        if self.height is not None and self.weight is not None:
            height_m = self.height / 100
            if height_m > 0:
                self.bmi = self.weight / (height_m ** 2)
                return
        if self.bmi is None:
            self.bmi = None

    # Override __init__ to automatically calculate BMI when data is set
    def __init__(self, **kwargs):
        if 'bmi' in kwargs and 'height' not in kwargs and 'weight' not in kwargs:
            super().__init__(**kwargs)
            return
        super().__init__(**kwargs)
        self.calculate_bmi()

    def __repr__(self):
        return f'<PatientData for User {self.user_id}>'


class Clinic(db.Model):
    """Stores information about IVF clinics."""
    __tablename__ = 'clinic' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(255))
    contact_number = db.Column(db.String(20))
    phone = db.Column(db.String(20))
    website = db.Column(db.String(255))
    description = db.Column(db.Text)
    latitude = db.Column(db.Float)  # Latitude for map markers
    longitude = db.Column(db.Float)  # Longitude for map markers
    zip_code = db.Column(db.String(10))  # Added zip_code for search
    clinic_type = db.Column(db.String(50), default='IVF')  # Type of clinic (IVF, Fertility, Reproductive Health)

    # Relationship to doctors working at this clinic
    doctors = db.relationship('User', backref='clinic', lazy='dynamic') # References 'user' table via clinic_id foreign key

    def __repr__(self):
        return f'<Clinic {self.name}>'


class IVFCycle(db.Model):
    """
    Tracks a specific IVF treatment cycle from stimulation to outcome.
    """
    __tablename__ = 'ivf_cycle' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # References 'user' table

    protocol = db.Column(db.String(100)) # e.g., Antagonist, Agonist
    start_date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(50), default='stimulation') # stimulation, retrieval, transfer, wait, complete
    
    # Key Milestones
    egg_retrieval_date = db.Column(db.Date)
    num_eggs_retrieved = db.Column(db.Integer)
    fertilization_rate = db.Column(db.Float) # Percentage (0.0 to 1.0)
    num_embryos_day5 = db.Column(db.Integer)
    
    transfer_date = db.Column(db.Date)
    num_embryos_transferred = db.Column(db.Integer)
    outcome = db.Column(db.String(50)) # BFN, BFP, Miscarriage, Live Birth

    # Renaming original notes to be patient-specific
    patient_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to doctor's notes
    doctor_notes = db.relationship('CycleNote', backref='cycle', lazy='dynamic', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<IVFCycle {self.id} for Patient {self.patient_id} ({self.protocol})>'


class CycleNote(db.Model):
    """Stores notes made by doctors on a specific IVF cycle."""
    __tablename__ = 'cycle_note' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    cycle_id = db.Column(db.Integer, db.ForeignKey('ivf_cycle.id'), nullable=False) # References 'ivf_cycle' table
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # References 'user' table
    note_content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to the doctor who wrote the note
    doctor = db.relationship('User', backref=db.backref('authored_notes', lazy='dynamic'))

    def __repr__(self):
        return f'<CycleNote {self.id} for Cycle {self.cycle_id} by Doctor {self.doctor_id}>'


class WellnessLog(db.Model):
    """
    Daily log for patient-reported symptoms, mood, and lifestyle factors.
    """
    __tablename__ = 'wellness_log' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # FIX: Using date.today for non-deprecated date tracking # <-- FINAL FIX 
    date = db.Column(db.Date, default=date.today, nullable=False)
    # Ratings (1=Very Poor, 5=Excellent)
    mood_rating = db.Column(db.Integer) # 1-5
    stress_level = db.Column(db.Integer) # 1-5
    sleep_quality = db.Column(db.Integer) # 1-5
    energy_level = db.Column(db.Integer) # 1-5
    nutrition_score = db.Column(db.Integer) # 1-5
    
    # Metrics
    sleep_hours = db.Column(db.Float)
    exercise_minutes = db.Column(db.Integer, default=0)
    meditation_minutes = db.Column(db.Integer, default=0)
    water_intake = db.Column(db.Integer, default=0) # in ml or glasses
    
    # Booleans
    yoga_practiced = db.Column(db.Boolean, default=False)
    
    # Qualitative Data
    symptoms = db.Column(db.Text) # Free text about physical symptoms
    notes = db.Column(db.Text)
    mood_notes = db.Column(db.Text)
    stress_factors = db.Column(db.Text)
    sleep_notes = db.Column(db.Text)
    supplements_taken = db.Column(db.Text)

    # Meal Descriptions
    meal_breakfast = db.Column(db.Text)
    meal_lunch = db.Column(db.Text)
    meal_dinner = db.Column(db.Text)
    meal_snacks = db.Column(db.Text)

    # AI-detected emotion from mood_notes
    detected_emotion = db.Column(db.String(50))

    # Ensures a user can only log one entry per day
    __table_args__ = (db.UniqueConstraint('user_id', 'date', name='_user_date_uc'),)

    def __repr__(self):
        return f'<WellnessLog {self.date} for User {self.user_id}>'


class MedicationReminder(db.Model):
    """
    Stores medication schedules for reminders and tracking.
    """
    __tablename__ = 'medication_reminder' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    medication_name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50))
    frequency = db.Column(db.String(50)) # e.g., Daily, Twice Daily
    time_of_day = db.Column(db.Time) # Time of the day for the reminder
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date) # Null if ongoing

    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<MedicationReminder {self.medication_name} for User {self.user_id}>'


class ChatMessage(db.Model):
    """
    Stores chat messages between users and the AI chatbot.
    """
    __tablename__ = 'chat_message' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<ChatMessage {self.id} for User {self.user_id}>'


class MedicalDocument(db.Model):
    """
    Stores metadata for uploaded medical documents.
    Upgraded for Centralized Medical Record Management.
    """
    __tablename__ = 'medical_document' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    filename = db.Column(db.String(255), nullable=False) # Stored filename
    original_filename = db.Column(db.String(255), nullable=False) # Original filename
    file_type = db.Column(db.String(100)) # MIME type
    file_size = db.Column(db.Integer) # Size in bytes
    description = db.Column(db.Text)
    extracted_text = db.Column(db.Text) # Extracted text from OCR/PDF processing
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # --- New fields for Centralized Medical Record System ---
    title = db.Column(db.String(255))  # User-friendly document title
    category = db.Column(db.String(50), default='other')  # lab_report, scan, prescription, other
    version = db.Column(db.Integer, default=1)  # Version number
    uploaded_by = db.Column(db.String(10), default='patient')  # patient, doctor
    tags = db.Column(db.String(255))  # Comma-separated tags for search

    # Idempotent OCR and report-freshness metadata. All columns are nullable or
    # have safe defaults so the existing document rows migrate without loss.
    file_hash = db.Column(db.String(64), index=True)
    report_date = db.Column(db.Date)
    processing_status = db.Column(db.String(32), default='UPLOADED')
    processed_at = db.Column(db.DateTime)
    is_latest = db.Column(db.Boolean, default=False, nullable=False)
    extracted_features = db.Column(db.Text)
    validation_summary = db.Column(db.Text)

    def __repr__(self):
        return f'<MedicalDocument {self.filename} (v{self.version}) for User {self.user_id}>'


class MedicalFeatureHistory(db.Model):
    """Append-only source/audit history for OCR-populated prediction features."""
    __tablename__ = 'medical_feature_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    document_id = db.Column(db.Integer, db.ForeignKey('medical_document.id'), nullable=True, index=True)
    feature_name = db.Column(db.String(40), nullable=False, index=True)
    old_value = db.Column(db.Float)
    new_value = db.Column(db.Float)
    report_date = db.Column(db.Date)
    extraction_source = db.Column(db.String(32), nullable=False, default='OCR')
    validation_status = db.Column(db.String(32), nullable=False)
    validation_message = db.Column(db.Text)
    confidence = db.Column(db.Float)  # Nullable: current OCR engine exposes no confidence score.
    extracted_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    user_modified = db.Column(db.Boolean, nullable=False, default=False)

    document = db.relationship('MedicalDocument', backref=db.backref('feature_history', lazy='dynamic'))

    def __repr__(self):
        return f'<MedicalFeatureHistory {self.feature_name} for document {self.document_id}>'


class Prediction(db.Model):
    """
    Stores the results of the AI model's success prediction and recommendation.
    """
    __tablename__ = 'prediction' # <-- CRITICAL FIX: Explicit table name added
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    prediction_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # AI Results
    success_probability = db.Column(db.Float) # Score (0.0 to 1.0)

    # Personalized Protocol Recommendation
    protocol_recommendation = db.Column(db.String(100))

    # Detailed text output from the LLM based on grounded search/analysis
    llm_analysis = db.Column(db.Text)

    # Metadata for the prediction (e.g., features used, model version)
    model_metadata = db.Column(db.Text)

    def __repr__(self):
        return f'<Prediction {self.id} ({self.success_probability:.2f}) for User {self.user_id}>'


class MedicalActivity(db.Model):
    """
    Stores medical activities performed on patients (injections, scans, blood work, etc.).
    """
    __tablename__ = 'medical_activity'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    activity_type = db.Column(db.String(50), nullable=False)  # injection, scan, bloodwork, consultation, embryo_transfer, egg_retrieval
    activity_name = db.Column(db.String(255), nullable=False)
    dosage = db.Column(db.String(100))  # For medications/injections
    performed_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    results = db.Column(db.Text)  # Test results or observations
    notes = db.Column(db.Text)

    # Relationship to patient
    patient = db.relationship('User', backref='medical_activities')

    def __repr__(self):
        return f'<MedicalActivity {self.activity_type}: {self.activity_name} for Patient {self.patient_id}>'


class Message(db.Model):
    """
    Stores messages between patients and doctors for communication.
    """
    __tablename__ = 'message'
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_messages')

    def __repr__(self):
        return f'<Message from {self.sender_id} to {self.receiver_id}>'


class Notification(db.Model):
    """
    Stores smart notifications and alerts for users (patients, doctors, admins).
    """
    __tablename__ = 'notification'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(20), default='info')  # info, warning, critical
    category = db.Column(db.String(30), default='system')  # medication, risk, wellness, appointment, system
    is_read = db.Column(db.Boolean, default=False)
    link = db.Column(db.String(255))  # optional URL to redirect when clicked
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to user
    user = db.relationship('User', backref=db.backref('notifications', lazy='dynamic', cascade="all, delete-orphan"))

    def __repr__(self):
        return f'<Notification {self.type} for User {self.user_id}: {self.message[:50]}>'


class DocumentVersion(db.Model):
    """
    Tracks version history for uploaded medical documents.
    """
    __tablename__ = 'document_version'
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('medical_document.id'), nullable=False)
    version = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    change_note = db.Column(db.Text)

    # Relationship to parent document
    document = db.relationship('MedicalDocument', backref=db.backref('versions', lazy='dynamic', cascade="all, delete-orphan"))

    def __repr__(self):
        return f'<DocumentVersion {self.version} of Document {self.document_id}>'


class DocumentAnnotation(db.Model):
    """
    Allows doctors to annotate/comment on medical documents.
    """
    __tablename__ = 'document_annotation'
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('medical_document.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    annotation_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    document = db.relationship('MedicalDocument', backref=db.backref('annotations', lazy='dynamic', cascade="all, delete-orphan"))
    doctor = db.relationship('User', backref=db.backref('document_annotations', lazy='dynamic'))

    def __repr__(self):
        return f'<DocumentAnnotation by Doctor {self.doctor_id} on Document {self.document_id}>'


class DocumentShare(db.Model):
    """
    Tracks sharing permissions for medical documents across doctors/clinics.
    """
    __tablename__ = 'document_share'
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('medical_document.id'), nullable=False)
    shared_with_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # doctor who can view
    shared_by_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)    # patient who shared
    shared_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    can_view = db.Column(db.Boolean, default=True)

    # Relationships
    document = db.relationship('MedicalDocument', backref=db.backref('shares', lazy='dynamic', cascade="all, delete-orphan"))
    shared_with = db.relationship('User', foreign_keys=[shared_with_user_id], backref=db.backref('shared_documents', lazy='dynamic'))
    shared_by = db.relationship('User', foreign_keys=[shared_by_user_id], backref=db.backref('documents_shared', lazy='dynamic'))

    def __repr__(self):
        return f'<DocumentShare Doc:{self.document_id} with User:{self.shared_with_user_id}>'


class AuditLog(db.Model):
    """
    Tracks user actions for security auditing and compliance (HIPAA/GDPR readiness).
    """
    __tablename__ = 'audit_log'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Nullable for login attempts/system actions
    action = db.Column(db.String(50), nullable=False) # e.g., 'LOGIN', 'VIEW_PATIENT', 'EXPORT_DATA'
    details = db.Column(db.String(255)) # Specific details about the action
    ip_address = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<AuditLog {self.action} by {self.user_id} at {self.timestamp}>'


class SimulationResult(db.Model):
    """
    Stores IVF Success Improvement Simulator results for future reference.
    Tracks what-if scenarios for lifestyle modifications and their predicted impact.
    """
    __tablename__ = 'simulation_result'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # Input features snapshot (JSON) — the base patient data used for simulation
    input_features = db.Column(db.Text, nullable=False)

    # Current prediction probability (baseline)
    current_probability = db.Column(db.Float, nullable=False)

    # Full simulation results (JSON array of improvement scenarios)
    simulation_results = db.Column(db.Text, nullable=False)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to user
    user = db.relationship('User', backref=db.backref('simulation_results', lazy='dynamic', cascade="all, delete-orphan"))

    def __repr__(self):
        return f'<SimulationResult {self.id} for User {self.user_id}: Baseline {self.current_probability:.1f}%>'


class PartnerConnection(db.Model):
    __tablename__ = 'partner_connection'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    partner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    accepted_at = db.Column(db.DateTime)
    disconnected_at = db.Column(db.DateTime)

    permissions = db.relationship('PartnerSharingPermission', backref='connection', uselist=False, cascade='all, delete-orphan')
    tasks = db.relationship('SharedTask', backref='connection', lazy='dynamic', cascade='all, delete-orphan')
    notes = db.relationship('SharedNote', backref='connection', lazy='dynamic', cascade='all, delete-orphan')
    checkins = db.relationship('PartnerCheckIn', backref='connection', lazy='dynamic', cascade='all, delete-orphan')
    messages = db.relationship('PartnerMessage', backref='connection', lazy='dynamic', cascade='all, delete-orphan')

    @property
    def patient(self):
        return User.query.get(self.patient_id)

    @property
    def partner(self):
        return User.query.get(self.partner_id)


class PartnerInvitation(db.Model):
    __tablename__ = 'partner_invitation'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    partner_email = db.Column(db.String(120), nullable=False)
    partner_name = db.Column(db.String(120), nullable=False)
    relationship = db.Column(db.String(30), nullable=False)
    token_hash = db.Column(db.String(255), nullable=False, unique=True)
    status = db.Column(db.String(20), default='pending')
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    accepted_at = db.Column(db.DateTime)
    declined_at = db.Column(db.DateTime)


class PartnerSharingPermission(db.Model):
    __tablename__ = 'partner_sharing_permission'
    id = db.Column(db.Integer, primary_key=True)
    connection_id = db.Column(db.Integer, db.ForeignKey('partner_connection.id'), nullable=False, unique=True)
    treatment_shared = db.Column(db.Boolean, default=True)
    appointments_shared = db.Column(db.Boolean, default=True)
    medications_shared = db.Column(db.Boolean, default=True)
    wellness_shared = db.Column(db.Boolean, default=False)
    mood_shared = db.Column(db.Boolean, default=False)
    nutrition_shared = db.Column(db.Boolean, default=True)
    documents_shared = db.Column(db.Boolean, default=False)
    doctor_notes_shared = db.Column(db.Boolean, default=False)
    messages_enabled = db.Column(db.Boolean, default=True)
    tasks_enabled = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class SharedTask(db.Model):
    __tablename__ = 'shared_task'
    id = db.Column(db.Integer, primary_key=True)
    connection_id = db.Column(db.Integer, db.ForeignKey('partner_connection.id'), nullable=False)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text)
    assigned_to = db.Column(db.String(20), default='partner')
    due_date = db.Column(db.Date)
    priority = db.Column(db.String(20), default='medium')
    status = db.Column(db.String(20), default='pending')
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)

    creator = db.relationship('User', foreign_keys=[created_by], backref='created_shared_tasks')


class SharedNote(db.Model):
    __tablename__ = 'shared_note'
    id = db.Column(db.Integer, primary_key=True)
    connection_id = db.Column(db.Integer, db.ForeignKey('partner_connection.id'), nullable=False)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    creator = db.relationship('User', foreign_keys=[created_by], backref='created_shared_notes')


class PartnerCheckIn(db.Model):
    __tablename__ = 'partner_check_in'
    id = db.Column(db.Integer, primary_key=True)
    connection_id = db.Column(db.Integer, db.ForeignKey('partner_connection.id'), nullable=False)
    checkin_type = db.Column(db.String(50), nullable=False)
    note = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class PartnerMessage(db.Model):
    __tablename__ = 'partner_message'
    id = db.Column(db.Integer, primary_key=True)
    connection_id = db.Column(db.Integer, db.ForeignKey('partner_connection.id'), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_partner_messages')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_partner_messages')


class PartnerNotification(db.Model):
    __tablename__ = 'partner_notification'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    partner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    type = db.Column(db.String(30), default='shared_update')
    title = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<PartnerNotification {self.title} for {self.user_id}>'
