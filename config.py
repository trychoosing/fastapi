from celery import Celery

CELERY_BROKER_URL = "redis://localhost:6379/0"  # Redis as message broker
CELERY_RESULT_BACKEND = "redis://localhost:6379/1" # Redis as result backend

celery_app = Celery(
    "fastapi_celery_example",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)