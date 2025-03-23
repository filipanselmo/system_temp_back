import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'  # тестовое название БД
    SQLALCHEMY_TRACK_MODIFICATIONS = False