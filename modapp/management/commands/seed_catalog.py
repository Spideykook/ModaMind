import os
import glob
import shutil
from django.core.management.base import BaseCommand
from django.core.files import File
from django.conf import settings
from modapp.models import Category, ClothingItem

class Command(BaseCommand):
    help = 'Seeds the database with generated fashion catalog images from the artifact directory.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--source-dir',
            type=str,
            default=r'C:\Users\KIIT0001\.gemini\antigravity-cli\brain\70513276-b034-4ae6-9828-995ca791dd89',
            help='Directory containing the generated images'
        )

    def handle(self, *args, **options):
        source_dir = options['source_dir']
        self.stdout.write(self.style.WARNING(f"Reading generated images from: {source_dir}"))

        if not os.path.exists(source_dir):
            self.stdout.write(self.style.ERROR(f"Source directory does not exist: {source_dir}"))
            return

        # Define categories to seed
        categories_data = {
            'outerwear': 'Outerwear',
            'tops': 'Tops',
            'bottoms': 'Bottoms',
            'dresses': 'Dresses',
            'shoes': 'Shoes',
        }

        categories = {}
        for slug, name in categories_data.items():
            category, created = Category.objects.get_or_create(slug=slug, defaults={'name': name})
            categories[slug] = category
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created category: {name}"))
            else:
                self.stdout.write(self.style.SUCCESS(f"Loaded existing category: {name}"))

        # Define items to seed
        # Each item has a search pattern key, target filename, display name, category slug, brand, and color
        items_to_seed = [
            {
                'pattern': 'black_leather_jacket_*.jpg',
                'filename': 'black_leather_jacket.jpg',
                'name': 'Classic Black Leather Biker Jacket',
                'category_slug': 'outerwear',
                'brand': 'Zara',
                'color': 'Black',
            },
            {
                'pattern': 'blue_denim_jacket_*.jpg',
                'filename': 'blue_denim_jacket.jpg',
                'name': 'Vintage Light Blue Denim Jacket',
                'category_slug': 'outerwear',
                'brand': "Levi's",
                'color': 'Blue',
            },
            {
                'pattern': 'beige_trench_coat_*.jpg',
                'filename': 'beige_trench_coat.jpg',
                'name': 'Double-Breasted Beige Trench Coat',
                'category_slug': 'outerwear',
                'brand': 'Burberry',
                'color': 'Beige',
            },
            {
                'pattern': 'red_hoodie_*.jpg',
                'filename': 'red_hoodie.jpg',
                'name': 'Red Classic Cotton Hoodie',
                'category_slug': 'tops',
                'brand': 'Champion',
                'color': 'Red',
            },
            {
                'pattern': 'white_tshirt_*.jpg',
                'filename': 'white_tshirt.jpg',
                'name': 'White Organic Cotton Tee',
                'category_slug': 'tops',
                'brand': 'Uniqlo',
                'color': 'White',
            },
            {
                'pattern': 'blue_jeans_*.jpg',
                'filename': 'blue_jeans.jpg',
                'name': '501 Original Fit Blue Jeans',
                'category_slug': 'bottoms',
                'brand': "Levi's",
                'color': 'Blue',
            },
            {
                'pattern': 'black_trousers_*.jpg',
                'filename': 'black_trousers.jpg',
                'name': 'Formal Tailored Black Trousers',
                'category_slug': 'bottoms',
                'brand': 'Hugo Boss',
                'color': 'Black',
            },
            {
                'pattern': 'yellow_summer_dress_*.jpg',
                'filename': 'yellow_summer_dress.jpg',
                'name': 'Yellow Floral Summer Dress',
                'category_slug': 'dresses',
                'brand': 'H&M',
                'color': 'Yellow',
            },
            {
                'pattern': 'white_sneakers_*.jpg',
                'filename': 'white_sneakers.jpg',
                'name': 'Minimalist White Leather Sneakers',
                'category_slug': 'shoes',
                'brand': 'Adidas',
                'color': 'White',
            },
            {
                'pattern': 'brown_leather_boots_*.jpg',
                'filename': 'brown_leather_boots.jpg',
                'name': 'Classic Brown Chelsea Boots',
                'category_slug': 'shoes',
                'brand': 'Clarks',
                'color': 'Brown',
            },
        ]

        for item_data in items_to_seed:
            pattern_path = os.path.join(source_dir, item_data['pattern'])
            matching_files = glob.glob(pattern_path)
            
            if not matching_files:
                self.stdout.write(self.style.WARNING(f"No file found matching pattern: {item_data['pattern']}"))
                continue

            # Pick the first match (most recent or only one)
            source_file_path = matching_files[0]
            
            # Check if item already exists in DB
            clothing_item, created = ClothingItem.objects.get_or_create(
                name=item_data['name'],
                defaults={
                    'category': categories[item_data['category_slug']],
                    'brand': item_data['brand'],
                    'color': item_data['color'],
                }
            )

            # If it's newly created or has no image, save the image
            if created or not clothing_item.image:
                with open(source_file_path, 'rb') as f:
                    django_file = File(f)
                    clothing_item.image.save(item_data['filename'], django_file, save=True)
                self.stdout.write(self.style.SUCCESS(f"Created & seeded clothing item: {item_data['name']}"))
            else:
                self.stdout.write(self.style.WARNING(f"Item already exists: {item_data['name']}"))
        
        self.stdout.write(self.style.SUCCESS("Catalog seeding complete! Run 'python manage.py build_index' to rebuild the search index."))
