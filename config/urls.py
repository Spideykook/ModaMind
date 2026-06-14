"""
Root URL configuration for the ModaMind project.

Delegates all non-admin routes to 'modapp.urls'. In DEBUG mode, also wires
up serving of user-uploaded MEDIA files (clothing images) so the frontend
can render <img src="..."> tags pointing at /media/clothing_images/...
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("modapp.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
