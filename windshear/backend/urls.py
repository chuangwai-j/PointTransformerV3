from django.urls import path
from .views import PredictAPI

urlpatterns = [
    path('predict/', PredictAPI.as_view(), name='predict'),
]
