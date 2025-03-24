from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return str(self.id)


class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String, nullable=False)
    content = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    shelf_id = db.Column(db.Integer, db.ForeignKey('shelf.id', name='fk_photo_shelf'), nullable=False)
    shelf = db.relationship('Shelf', backref='photos')
    # shelf_id = db.Column(db.String, nullable=False, default='default_shelf')
    compliance_checks = db.relationship('ComplianceCheckResult', backref='photo', lazy=True)
    embeddings = db.relationship('Embedding', backref='photo', lazy=True)


class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    photo_id = db.Column(db.Integer, db.ForeignKey('photo.id', name='fk_planogram_shelf'), nullable=False)
    label = db.Column(db.String, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    x_min = db.Column(db.Float, nullable=False)
    y_min = db.Column(db.Float, nullable=False)
    x_max = db.Column(db.Float, nullable=False)
    y_max = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Связь с моделью Photo
    photo = db.relationship('Photo', backref=db.backref('detection_results', lazy=True))

# Модель для планограммы
class Planogram(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # shelf_id = db.Column(db.String, nullable=False)
    shelf_id = db.Column(db.Integer, db.ForeignKey('shelf.id'), nullable=False)
    shelf = db.relationship('Shelf', backref='planograms')
    sku = db.Column(db.String, nullable=False)
    x_min = db.Column(db.Float, nullable=False)
    y_min = db.Column(db.Float, nullable=False)
    x_max = db.Column(db.Float, nullable=False)
    y_max = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# Модель для результатов сверки
class ComplianceCheckResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    photo_id = db.Column(db.Integer, db.ForeignKey('photo.id'), nullable=False)
    sku = db.Column(db.String, nullable=False)
    missing_count = db.Column(db.Integer, nullable=False)
    extra_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())


class Embedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    photo_id = db.Column(db.Integer, db.ForeignKey('photo.id'), nullable=False)
    features = db.Column(db.LargeBinary, nullable=False)

# Добавляем модель категории
class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    shelves = db.relationship('Shelf', backref='category', lazy=True)

# Добавляем модель стеллажа
class Shelf(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    identifier = db.Column(db.String(50), nullable=False, unique=True)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# Добавляем модель задания
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    shelf_id = db.Column(db.Integer, db.ForeignKey('shelf.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    shelf = db.relationship('Shelf', backref='tasks')