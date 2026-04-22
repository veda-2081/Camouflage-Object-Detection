import os


class Config:

    SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")

    # upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    UPLOAD_FOLDER = "uploads"

    # model settings
    IMG_SIZE = 640

    # flask debug
    DEBUG = True