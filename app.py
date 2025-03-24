import base64
import logging
import os
import uuid
from sqlite3 import IntegrityError

from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision import transforms
from flask import Flask, render_template, send_from_directory, request, jsonify,redirect, url_for,flash
from flask_cors import CORS
from io import BytesIO
from OptimizedYOLO import OptimizedYOLO
from dbModels import db, DetectionResult, Photo, Planogram, ComplianceCheckResult, Embedding, Task, Shelf, Category, \
    User
from forms import LoginForm, AdminUserForm
from gan import augment_image
from triplet_net import TripletNet
from flask_migrate import Migrate
from flask_login import LoginManager, login_user, logout_user, login_required, current_user


app = Flask(__name__)
app.config.from_object('config.Config')
CORS(app)
db.init_app(app)
migrate = Migrate(app, db)
yolo = OptimizedYOLO()
triplet_net = TripletNet()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('task_list'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('task_list'))
        flash('Неверный логин или пароль')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        os.abort(403)
    users = User.query.all()
    return render_template('admin/dashboard.html', users=users)

@app.route('/admin/create-user', methods=['GET', 'POST'])
@login_required
def create_user():
    if not current_user.is_admin:
        os.abort(403)

    form = AdminUserForm()
    if form.validate_on_submit():
        try:
            user = User(
                username=form.username.data,
                is_admin=form.is_admin.data,
                is_active=True
            )
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Пользователь создан', 'success')
            return redirect(url_for('admin_dashboard'))
        except IntegrityError:
            db.session.rollback()
            flash('Имя пользователя уже существует', 'danger')
    return render_template('admin/create_user.html', form=form)


# Расчет Intersection over Union
def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def generate_recommendations(compliance_checks):
    recommendations = []
    for check in compliance_checks:
        if check.missing_count > 0:
            recommendations.append({
                'sku': check.sku,
                'action': 'restock',
                'quantity': check.missing_count,
                'priority': 'high' if check.missing_count > 2 else 'medium'
            })
        if check.extra_count > 0:
            recommendations.append({
                'sku': check.sku,
                'action': 'remove',
                'quantity': check.extra_count,
                'priority': 'low'
            })
    return recommendations


def draw_boxes(photo):
    try:
        image = Image.open(BytesIO(photo.content))
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # Отрисовка планограммы
        planogram_entries = Planogram.query.filter_by(shelf_id=photo.shelf_id).all()
        for entry in planogram_entries:
            x_min = entry.x_min * width
            y_min = entry.y_min * height
            x_max = entry.x_max * width
            y_max = entry.y_max * height
            draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)
            draw.text((x_min, y_min - 15), f"Plan: {entry.sku}", fill="blue")

        # Отрисовка детекций
        detections = DetectionResult.query.filter_by(photo_id=photo.id).all()
        for det in detections:
            x_min = det.x_min * width
            y_min = det.y_min * height
            x_max = det.x_max * width
            y_max = det.y_max * height
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
            draw.text((x_min, y_max + 5), f"{det.label} {det.confidence:.2f}", fill="green")

        # Отрисовка отклонений
        compliance_checks = ComplianceCheckResult.query.filter_by(photo_id=photo.id).all()
        for check in compliance_checks:
            if check.missing_count > 0:
                entry = Planogram.query.filter_by(shelf_id=photo.shelf_id, sku=check.sku).first()
                if entry:
                    x_min = entry.x_min * width
                    y_min = entry.y_min * height
                    x_max = entry.x_max * width
                    y_max = entry.y_max * height
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                    draw.text((x_min, y_min - 30), f"Missing: {check.missing_count}", fill="red")

        buf = BytesIO()
        image.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error drawing boxes: {str(e)}")
        return None


@app.route('/')
@login_required
def home():
    if current_user.is_authenticated:
        return redirect(url_for('task_list'))
    return redirect(url_for('login'))  # Перенаправляем на вход

@app.route('/results')
@login_required
def results():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    paged_results = DetectionResult.query.join(Photo).order_by(Photo.created_at.desc()).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    photos = Photo.query.all()
    return render_template('index.html', photos=photos, paged_results=paged_results)


# Добавляем новый маршрут для загрузки файлов из папки 'uploads'
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)


@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    # shelf_id = request.form.get('shelf_id')
    shelf_id = 1
    if not shelf_id:
        return jsonify({'error': 'shelf_id is required'}), 400

    # Сохранение файла
    filename = f"{uuid.uuid4()}_{file.filename}"
    image = Image.open(file.stream)
    # image = Image.open(file.stream).convert('L')
    augmented_image = augment_image(image)
    img_byte_arr = BytesIO()
    augmented_image.save(img_byte_arr, format='PNG')
    print("144")
    # img_array = np.array(augmented_image)[:, :, ::-1]
    img_array = np.array(augmented_image)

    photo = Photo(
        filename=filename,
        content=img_byte_arr.getvalue(),
        shelf_id=shelf_id
    )
    print("152")
    db.session.add(photo)
    db.session.commit()

    # # Детектирование
    # # detections = yolo.detect(augmented_image)
    detections = yolo.detect(img_array)
    # print("159")
    # for det in detections:
    #     detection = DetectionResult(
    #         photo_id=photo.id,
    #         label=det['class'],
    #         confidence=det['confidence'],
    #         x_min=det['bbox'][0],
    #         y_min=det['bbox'][1],
    #         x_max=det['bbox'][2],
    #         y_max=det['bbox'][3]
    #     )
    #     db.session.add(detection)
    # db.session.commit()

    # Извлекаем первый тензор (для одного изображения)
    detections = detections[0]  # detections - список тензоров

    if detections is not None:
        for det in detections:
            # Распаковываем тензор
            x_min, y_min, x_max, y_max, conf, cls = det.tolist()

            detection = DetectionResult(
                photo_id=photo.id,
                label=int(cls),  # Класс как целое число
                confidence=float(conf),  # Доверие как float
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max
            )
            db.session.add(detection)
    else:
        print("Нет обнаружений")
    db.session.commit()

    # Генерация эмбеддингов
    img_tensor = transforms.ToTensor()(augmented_image).unsqueeze(0)
    print("175")
    with torch.no_grad():
        embedding = triplet_net(img_tensor).numpy()
        print("178")
    emb = Embedding(photo_id=photo.id, features=embedding.tobytes())
    db.session.add(emb)
    db.session.commit()

    # Проверка соответствия планограмме
    check_compliance(photo.id, shelf_id)
    print("185")
    return jsonify({'message': 'Success'}), 201


def check_compliance(photo_id, shelf_id):
    # Получение планограммы для полки
    print("191")
    planogram_entries = Planogram.query.filter_by(shelf_id=shelf_id).all()
    print("193")
    detected_boxes = DetectionResult.query.filter_by(photo_id=photo_id).all()

    sku_stats = {}
    for entry in planogram_entries:
        sku = entry.sku
        expected_quantity = entry.quantity
        sku_stats[sku] = {
            'expected': expected_quantity,
            'detected': 0,
            'missing': 0,
            'extra': 0
        }

    # Подсчет обнаруженных товаров
    for detection in detected_boxes:
        matched = False
        for entry in planogram_entries:
            if detection.label == entry.sku:
                iou = calculate_iou(
                    [detection.x_min, detection.y_min, detection.x_max, detection.y_max],
                    [entry.x_min, entry.y_min, entry.x_max, entry.y_max]
                )
                if iou > 0.5:  # Порог IoU для совпадения
                    sku_stats[entry.sku]['detected'] += 1
                    matched = True
                    break
        if not matched:
            # Лишний товар
            compliance_result = ComplianceCheckResult(
                photo_id=photo_id,
                sku=detection.label,
                missing_count=0,
                extra_count=1
            )
            db.session.add(compliance_result)

    # Вычисление недостачи
    for sku, stats in sku_stats.items():
        missing = stats['expected'] - stats['detected']
        if missing > 0:
            compliance_result = ComplianceCheckResult(
                photo_id=photo_id,
                sku=sku,
                missing_count=missing,
                extra_count=0
            )
            db.session.add(compliance_result)

    db.session.commit()


@app.route('/photos', methods=['GET'])
def get_photos():
    photos = Photo.query.all()
    results = []

    for photo in photos:
        print("275")
        # annotated_image = draw_boxes(photo)
        print("277")
        compliance_checks = ComplianceCheckResult.query.filter_by(photo_id=photo.id).all()
        compliance_info = [{
            'sku': check.sku,
            'missing': check.missing_count,
            'extra': check.extra_count
        } for check in compliance_checks]
        print("284")

        recommendations = generate_recommendations(compliance_checks)

        photo_data = {
            'id': photo.id,
            'filename': photo.filename,
            'shelf_id': photo.shelf_id,
            # 'annotated_image': annotated_image,
            'compliance_info': compliance_info,
            'recommendations': recommendations
        }
        print("296")
        results.append(photo_data)
        print(results)

    return jsonify(results), 200


@app.route('/tasks')
@login_required
def task_list():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    tasks_pagination = Task.query.order_by(Task.created_at.desc()).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    return render_template('tasks.html',
        tasks=tasks_pagination.items,
        pagination=tasks_pagination
    )

@app.route('/task/<int:task_id>')
@login_required
def task_detail(task_id):
    task = Task.query.get_or_404(task_id)
    shelf = task.shelf
    category = shelf.category if shelf else None
    return render_template('task_detail.html', task=task, shelf=shelf, category=category)

@app.route('/create-task', methods=['GET', 'POST'])
@login_required
def create_task():
    if request.method == 'POST':
        # Обработка существующего стеллажа
        shelf_id = request.form.get('shelf_id')
        new_identifier = request.form.get('new_shelf_identifier')

        # Обработка категорий
        category_type = request.form.get('category_type')
        category_id = None

        if category_type == 'existing':
            category_id = request.form.get('existing_category')
        else:
            new_category_name = request.form.get('new_category_name').strip()
            if new_category_name:
                # Проверяем существование категории
                category = Category.query.filter_by(name=new_category_name).first()
                if not category:
                    category = Category(name=new_category_name)
                    db.session.add(category)
                    db.session.commit()
                category_id = category.id

        # Валидация
        if new_identifier and not category_id:
            return "Выберите или создайте категорию", 400

        # Создание стеллажа при необходимости
        if new_identifier:
            shelf = Shelf(
                identifier=new_identifier,
                category_id=category_id
            )
            db.session.add(shelf)
            db.session.commit()
            shelf_id = shelf.id
        else:
            shelf_id = int(shelf_id)

        # Создание задания
        task = Task(shelf_id=shelf_id)
        db.session.add(task)
        db.session.commit()

        return redirect('/tasks')

    # GET-запрос
    shelves = Shelf.query.all()
    categories = Category.query.all()
    return render_template('create_task.html', shelves=shelves, categories=categories)

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    tasks = Task.query.filter_by(status='pending').all()
    return jsonify([{
        'id': t.id,
        'shelf_id': t.shelf_id,
        'shelf_identifier': t.shelf.identifier,
        'category': t.shelf.category.name,
        'created_at': t.created_at.isoformat()
    } for t in tasks])

@app.route('/detection-results', methods=['GET'])
def get_detection_results():
    results = DetectionResult.query.all()
    return jsonify([
        {
            'id': result.id,
            'photo_id': result.photo_id,
            'label': result.label,
            'confidence': result.confidence,
            'x_min': result.x_min,
            'y_min': result.y_min,
            'x_max': result.x_max,
            'y_max': result.y_max,
            'created_at': result.created_at.isoformat(),
        }
        for result in results
    ]), 200


@app.route('/photos-with-results', methods=['GET'])
def get_photos_with_results():
    photos = Photo.query.all()
    results = DetectionResult.query.all()

    # Группируем результаты обнаружения по фотографиям
    grouped_results = {}
    for result in results:
        if result.photo_id not in grouped_results:
            grouped_results[result.photo_id] = []
        grouped_results[result.photo_id].append({
            'id': result.id,
            'label': result.label,
            'confidence': result.confidence,
            'x_min': result.x_min,
            'y_min': result.y_min,
            'x_max': result.x_max,
            'y_max': result.y_max,
            'created_at': result.created_at.isoformat(),
        })

    # Формируем ответ
    response = []
    for photo in photos:
        content = base64.b64encode(photo.content).decode('ascii')
        # Проверяем наличие фото перед добавлением в ответ
        if photo is not None:
            response.append({
                'id': photo.id,
                'filename': photo.filename,
                'content': content,
                'results': grouped_results.get(photo.id, [])
            })
    return jsonify(response), 200


@app.errorhandler(500)
def internal_error(error):
    logging.exception("An error occurred: %s", error)
    return "Internal Server Error", 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        app.run(host='0.0.0.0', port=5000, debug=True)
