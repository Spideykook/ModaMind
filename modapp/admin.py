from django.contrib import admin

from .models import ClothingItem


@admin.register(ClothingItem)
class ClothingItemAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "category", "brand", "is_indexed", "created_at")
    list_filter = ("category", "is_indexed")
    search_fields = ("title", "brand", "description")
    readonly_fields = ("is_indexed", "created_at", "updated_at")
