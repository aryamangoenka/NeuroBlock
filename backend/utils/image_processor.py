import os
import numpy as np
import zipfile
import tarfile
import tempfile
import shutil
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Handles processing of image datasets from archived folders.
    Supports zip and tar archives with folder-based class organization.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), channels: int = 3):
        """
        Initialize the image processor.
        
        Args:
            target_size: Target size for all images (width, height)
            channels: Number of channels (1 for grayscale, 3 for RGB)
        """
        self.target_size = target_size
        self.channels = channels
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def is_archive_file(self, filename: str) -> bool:
        """Check if file is a supported archive format."""
        filename_lower = filename.lower()
        return (filename_lower.endswith('.zip') or 
                filename_lower.endswith('.tar') or 
                filename_lower.endswith('.tar.gz') or 
                filename_lower.endswith('.tgz'))
    
    def extract_archive(self, archive_path: str, extract_to: str) -> str:
        """
        Extract archive to specified directory.
        
        Args:
            archive_path: Path to the archive file
            extract_to: Directory to extract to
            
        Returns:
            Path to extracted contents
        """
        logger.info(f"Extracting archive: {archive_path}")
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
            
        return extract_to
    
    def find_image_folders(self, root_path: str) -> Dict[str, List[str]]:
        """
        Find folders containing images and map them to class names.
        
        Args:
            root_path: Root directory to search
            
        Returns:
            Dictionary mapping class names to lists of image file paths
        """
        class_folders = {}
        root = Path(root_path)
        
        # Recursively search for folders containing images
        def search_for_class_folders(path: Path, depth: int = 0, max_depth: int = 3):
            """Recursively search for folders containing images"""
            if depth > max_depth:
                return
                
            for item in path.iterdir():
                if item.is_dir():
                    # Check if this directory contains images directly
                    direct_images = []
                    for img_file in item.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in self.supported_formats:
                            direct_images.append(str(img_file))
                    
                    if direct_images:
                        # This folder contains images directly - treat as class folder
                        class_name = item.name
                        class_folders[class_name] = direct_images
                        logger.info(f"Found class '{class_name}' with {len(direct_images)} images")
                    else:
                        # This folder doesn't contain images directly, search deeper
                        search_for_class_folders(item, depth + 1, max_depth)
        
        # Start the recursive search
        search_for_class_folders(root)
        
        # If no class folders found, check if root contains images directly
        if not class_folders:
            image_files = []
            for img_file in root.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in self.supported_formats:
                    image_files.append(str(img_file))
            
            if image_files:
                class_folders['default'] = image_files
                logger.info(f"Found {len(image_files)} images in root directory (single class)")
        
        return class_folders
    
    def process_image(self, image_path: str) -> np.ndarray:
        """
        Process a single image to the target format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed image as numpy array
        """
        try:
            with Image.open(image_path) as img:
                # Handle EXIF orientation (auto-rotate based on metadata)
                try:
                    # Try to get ORIENTATION constant, fallback to numeric value
                    try:
                        from PIL.ExifTags import ORIENTATION
                        orientation_tag = ORIENTATION
                    except ImportError:
                        # Fallback to the numeric value for ORIENTATION
                        orientation_tag = 274
                    
                    exif = img._getexif()
                    if exif is not None:
                        orientation = exif.get(orientation_tag)
                        if orientation == 3:
                            img = img.rotate(180, expand=True)
                        elif orientation == 6:
                            img = img.rotate(270, expand=True)
                        elif orientation == 8:
                            img = img.rotate(90, expand=True)
                except (AttributeError, KeyError, TypeError, ImportError):
                    # No EXIF data or orientation info, continue normally
                    pass
                
                # Convert problematic color modes to standard formats
                if img.mode in ('CMYK', 'LAB'):
                    # Convert CMYK or LAB to RGB first
                    img = img.convert('RGB')
                elif img.mode == 'P':
                    # Convert palette images to RGB
                    img = img.convert('RGB')
                elif img.mode in ('LA', 'RGBA'):
                    # Handle images with alpha channel
                    if self.channels == 3:
                        # Create white background and paste image
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                        else:  # LA mode
                            background.paste(img.convert('RGB'))
                        img = background
                    else:
                        # Convert to grayscale, dropping alpha
                        img = img.convert('L')
                
                # Final conversion to target format
                if self.channels == 1:
                    img = img.convert('L')
                elif self.channels == 3:
                    img = img.convert('RGB')
                else:
                    raise ValueError(f"Unsupported number of channels: {self.channels}")
                
                # Resize image
                img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32)
                
                # Ensure correct shape
                if self.channels == 1 and len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, axis=-1)
                elif self.channels == 3 and len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                
                # Normalize to [0, 1]
                img_array = img_array / 255.0
                
                return img_array
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise
    
    def process_dataset(self, archive_path: str, dataset_name: str) -> Dict:
        """
        Process an entire image dataset from an archive.
        
        Args:
            archive_path: Path to the archive file
            dataset_name: Name for the dataset
            
        Returns:
            Dictionary containing processed dataset information
        """
        logger.info(f"Processing image dataset: {dataset_name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract archive
            extract_path = self.extract_archive(archive_path, temp_dir)
            
            # Find image folders
            class_folders = self.find_image_folders(extract_path)
            
            if not class_folders:
                raise ValueError("No images found in the archive")
            
            # Process images
            all_images = []
            all_labels = []
            class_names = sorted(class_folders.keys())
            
            logger.info(f"Found {len(class_names)} classes: {class_names}")
            
            for class_idx, class_name in enumerate(class_names):
                image_paths = class_folders[class_name]
                logger.info(f"Processing class '{class_name}': {len(image_paths)} images")
                
                for img_path in image_paths:
                    try:
                        processed_img = self.process_image(img_path)
                        all_images.append(processed_img)
                        all_labels.append(class_idx)
                    except Exception as e:
                        logger.warning(f"Skipping image {img_path}: {str(e)}")
                        continue
            
            if not all_images:
                raise ValueError("No images could be processed successfully")
            
            # Convert to numpy arrays
            X = np.array(all_images)
            y = np.array(all_labels)
            
            # Calculate dataset statistics
            total_images = len(all_images)
            images_per_class = {class_names[i]: int(np.sum(y == i)) for i in range(len(class_names))}
            
            # Create metadata
            metadata = {
                'name': dataset_name,
                'task_type': 'classification',
                'dataset_type': 'image',
                'target_column': 'class',
                'feature_columns': [],  # Images don't have named features
                'original_filename': os.path.basename(archive_path),
                'shape': [int(total_images), int(len(class_names))],
                'image_shape': [int(x) for x in X.shape[1:]],
                'processed_shape': [int(total_images), int(np.prod(X.shape[1:]))],
                'created_at': datetime.now().isoformat(),
                'class_labels': class_names,
                'feature_types': ['image'],
                'images_per_class': images_per_class,
                'total_images': int(total_images),
                'target_size': [int(x) for x in self.target_size],
                'channels': int(self.channels),
                'data_quality': self._assess_image_dataset_quality(images_per_class, total_images)
            }
            
            logger.info(f"Successfully processed {total_images} images from {len(class_names)} classes")
            
            return {
                'X': X,
                'y': y,
                'metadata': metadata
            }
    
    def _assess_image_dataset_quality(self, images_per_class: Dict[str, int], total_images: int) -> Dict:
        """Assess the quality of the image dataset."""
        warnings = []
        recommendations = []
        
        # Check dataset size
        if total_images < 100:
            warnings.append(f"Very small dataset ({total_images} images)")
            recommendations.append("Consider collecting more images for better model performance")
        elif total_images < 1000:
            warnings.append(f"Small dataset ({total_images} images)")
            recommendations.append("Consider data augmentation techniques")
        
        # Check class balance
        class_counts = list(images_per_class.values())
        if len(class_counts) > 1:
            min_count = min(class_counts)
            max_count = max(class_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 10:
                warnings.append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f})")
                recommendations.append("Consider class balancing techniques or data augmentation")
            elif imbalance_ratio > 3:
                warnings.append(f"Moderate class imbalance (ratio: {imbalance_ratio:.1f})")
                recommendations.append("Monitor training for class bias")
        
        # Check minimum images per class
        min_images_per_class = min(class_counts) if class_counts else 0
        if min_images_per_class < 10:
            warnings.append(f"Some classes have very few images (min: {min_images_per_class})")
            recommendations.append("Ensure each class has sufficient examples")
        
        # Determine quality score
        quality_score = 100
        quality_score -= len(warnings) * 15
        quality_score -= max(0, (10 - imbalance_ratio) * 5) if len(class_counts) > 1 else 0
        quality_score = max(0, min(100, quality_score))
        
        if quality_score >= 80:
            quality_level = "excellent"
        elif quality_score >= 60:
            quality_level = "good"
        elif quality_score >= 40:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            'quality_score': int(quality_score),
            'quality_level': quality_level,
            'warnings': warnings,
            'recommendations': recommendations,
            'analysis': {
                'n_samples': int(total_images),
                'n_classes': int(len(class_counts)),
                'task_type': 'classification',
                'min_images_per_class': int(min_images_per_class),
                'max_images_per_class': int(max(class_counts)) if class_counts else 0,
                'imbalance_ratio': float(imbalance_ratio) if len(class_counts) > 1 else 1.0
            }
        } 