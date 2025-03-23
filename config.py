import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'  # тестовое название БД
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.urandom(24).hex()