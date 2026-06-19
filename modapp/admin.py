from django.contrib import admin
from .models import Category, ClothingItem, EmbeddingMetadata


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("id", "slug", "name")
    search_fields = ("slug", "name")
    prepopulated_fields = {"slug": ("name",)}


class EmbeddingMetadataInline(admin.StackedInline):
    model = EmbeddingMetadata
    extra = 0
    readonly_fields = ("model_name", "embedding_version", "created_at")
    can_delete = False


@admin.register(ClothingItem)
class ClothingItemAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "display_category_col", "brand", "color", "is_indexed", "created_at")
    list_filter = ("category", "is_indexed")
    search_fields = ("name", "brand", "color")
    readonly_fields = ("is_indexed", "created_at", "updated_at")
    raw_id_fields = ("category",)
    inlines = [EmbeddingMetadataInline]

    @admin.display(description="Category")
    def display_category_col(self, obj):
        return obj.display_category


@admin.register(EmbeddingMetadata)
class EmbeddingMetadataAdmin(admin.ModelAdmin):
    list_display = ("id", "clothing_item", "model_name", "embedding_version", "created_at")
    list_filter = ("model_name", "embedding_version")
    readonly_fields = ("created_at",)
