from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ClothingItem",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "image",
                    models.ImageField(
                        help_text="The catalog photo used to generate the ResNet50 embedding.",
                        upload_to="clothing_images/",
                    ),
                ),
                ("title", models.CharField(blank=True, default="", max_length=255)),
                (
                    "category",
                    models.CharField(
                        blank=True,
                        choices=[
                            ("tops", "Tops"),
                            ("bottoms", "Bottoms"),
                            ("dresses", "Dresses"),
                            ("outerwear", "Outerwear"),
                            ("footwear", "Footwear"),
                            ("accessories", "Accessories"),
                        ],
                        default="",
                        max_length=32,
                    ),
                ),
                ("brand", models.CharField(blank=True, default="", max_length=100)),
                ("description", models.TextField(blank=True, default="")),
                (
                    "is_indexed",
                    models.BooleanField(
                        default=False,
                        help_text="True once this item's embedding has been added to the FAISS index.",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "Clothing Item",
                "verbose_name_plural": "Clothing Items",
                "ordering": ["-created_at"],
            },
        ),
    ]
