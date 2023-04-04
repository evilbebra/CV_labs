from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),

    path ('model.json',views.model, name = "model"),
    path ('metadata.json', views.metadata, name = "metadata"),
    path ('weights.bin', views.weights, name = "weights")
]
