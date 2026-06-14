from django.urls import path

from . import views

app_name = "modapp"

urlpatterns = [
    path("", views.IndexView.as_view(), name="index"),
    path("api/search/", views.SimilaritySearchView.as_view(), name="similarity-search"),
]
