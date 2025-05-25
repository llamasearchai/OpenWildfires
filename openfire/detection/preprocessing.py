"""
Advanced image preprocessing for fire and smoke detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import structlog

logger = structlog.get_logger(__name__)


class ImagePreprocessor:
    """Advanced image preprocessing for fire detection."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        enhance_contrast: bool = True,
        reduce_noise: bool = True
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.reduce_noise = reduce_noise
        
        # Initialize augmentation pipeline
        self.transform = self._create_transform_pipeline()
        
    def _create_transform_pipeline(self) -> A.Compose:
        """Create the image transformation pipeline."""
        transforms = []
        
        # Resize to target size
        transforms.append(
            A.Resize(
                height=self.target_size[0],
                width=self.target_size[1],
                interpolation=cv2.INTER_LINEAR
            )
        )
        
        # Enhance contrast if enabled
        if self.enhance_contrast:
            transforms.extend([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.8),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                )
            ])
        
        # Noise reduction
        if self.reduce_noise:
            transforms.append(
                A.GaussianBlur(blur_limit=(1, 3), p=0.2)
            )
        
        # Normalization
        if self.normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                )
            )
        
        return A.Compose(transforms)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process an image through the preprocessing pipeline."""
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image
                processed = image.copy()
            else:
                logger.warning("Unexpected image format, attempting to convert")
                processed = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Apply fire-specific enhancements
            processed = self._enhance_for_fire_detection(processed)
            
            # Apply transformation pipeline
            transformed = self.transform(image=processed)
            processed_image = transformed["image"]
            
            # Convert back to numpy if tensor
            if hasattr(processed_image, 'numpy'):
                processed_image = processed_image.numpy()
            
            # Ensure correct data type
            if processed_image.dtype != np.uint8 and not self.normalize:
                processed_image = (processed_image * 255).astype(np.uint8)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return resized original image as fallback
            return cv2.resize(image, self.target_size)
    
    def _enhance_for_fire_detection(self, image: np.ndarray) -> np.ndarray:
        """Apply fire-specific image enhancements."""
        # Convert to HSV for better fire detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Enhance red and orange channels (typical fire colors)
        enhanced = image.copy()
        
        # Boost red channel
        enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.1, 0, 255)
        
        # Create fire color mask to enhance fire regions
        fire_mask = self._create_fire_color_mask(hsv)
        
        # Apply enhancement only to potential fire regions
        enhanced_fire = cv2.addWeighted(enhanced, 0.7, image, 0.3, 0)
        enhanced = np.where(fire_mask[..., np.newaxis], enhanced_fire, enhanced)
        
        return enhanced.astype(np.uint8)
    
    def _create_fire_color_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Create a mask for fire-like colors."""
        # Define fire color ranges in HSV
        # Red-orange range
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Orange-yellow range
        lower_orange = np.array([10, 50, 50])
        upper_orange = np.array([25, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv_image, lower_orange, upper_orange)
        
        # Combine masks
        fire_mask = cv2.bitwise_or(mask1, mask2)
        fire_mask = cv2.bitwise_or(fire_mask, mask3)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        return fire_mask > 0


class SmokePreprocessor(ImagePreprocessor):
    """Specialized preprocessing for smoke detection."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True
    ):
        super().__init__(target_size, normalize, enhance_contrast=True, reduce_noise=False)
    
    def _enhance_for_fire_detection(self, image: np.ndarray) -> np.ndarray:
        """Override with smoke-specific enhancements."""
        return self._enhance_for_smoke_detection(image)
    
    def _enhance_for_smoke_detection(self, image: np.ndarray) -> np.ndarray:
        """Apply smoke-specific image enhancements."""
        # Convert to LAB color space for better smoke detection
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Enhance L channel (lightness) to better detect smoke
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Reconstruct LAB image
        lab_enhanced = lab.copy()
        lab_enhanced[:, :, 0] = l_enhanced
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply edge enhancement to detect smoke boundaries
        enhanced = self._enhance_edges(enhanced)
        
        return enhanced
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges to better detect smoke boundaries."""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert edges back to 3-channel
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend with original image
        enhanced = cv2.addWeighted(image, 0.8, edges_3ch, 0.2, 0)
        
        return enhanced


class ThermalPreprocessor:
    """Preprocessing for thermal/infrared images."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        temperature_range: Tuple[float, float] = (0.0, 100.0)
    ):
        self.target_size = target_size
        self.temperature_range = temperature_range
    
    def process(self, thermal_image: np.ndarray) -> np.ndarray:
        """Process thermal image for fire detection."""
        try:
            # Normalize temperature values
            normalized = self._normalize_temperature(thermal_image)
            
            # Apply thermal-specific enhancements
            enhanced = self._enhance_thermal(normalized)
            
            # Resize to target size
            resized = cv2.resize(enhanced, self.target_size)
            
            return resized
            
        except Exception as e:
            logger.error(f"Thermal preprocessing failed: {e}")
            return cv2.resize(thermal_image, self.target_size)
    
    def _normalize_temperature(self, thermal_image: np.ndarray) -> np.ndarray:
        """Normalize thermal image based on temperature range."""
        min_temp, max_temp = self.temperature_range
        
        # Clip values to expected range
        clipped = np.clip(thermal_image, min_temp, max_temp)
        
        # Normalize to 0-255 range
        normalized = ((clipped - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)
        
        return normalized
    
    def _enhance_thermal(self, thermal_image: np.ndarray) -> np.ndarray:
        """Apply thermal-specific enhancements."""
        # Apply histogram equalization
        equalized = cv2.equalizeHist(thermal_image)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # Convert to 3-channel for consistency
        if len(blurred.shape) == 2:
            enhanced = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        else:
            enhanced = blurred
        
        return enhanced


class MultiModalPreprocessor:
    """Preprocessor for multi-modal inputs (RGB + Thermal)."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640)
    ):
        self.target_size = target_size
        self.rgb_preprocessor = ImagePreprocessor(target_size)
        self.thermal_preprocessor = ThermalPreprocessor(target_size)
    
    def process(
        self,
        rgb_image: np.ndarray,
        thermal_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Process multi-modal input."""
        # Process RGB image
        processed_rgb = self.rgb_preprocessor.process(rgb_image)
        
        if thermal_image is not None:
            # Process thermal image
            processed_thermal = self.thermal_preprocessor.process(thermal_image)
            
            # Combine RGB and thermal
            combined = self._combine_modalities(processed_rgb, processed_thermal)
            return combined
        else:
            return processed_rgb
    
    def _combine_modalities(
        self,
        rgb_image: np.ndarray,
        thermal_image: np.ndarray
    ) -> np.ndarray:
        """Combine RGB and thermal images."""
        # Ensure both images have same dimensions
        if rgb_image.shape[:2] != thermal_image.shape[:2]:
            thermal_image = cv2.resize(thermal_image, rgb_image.shape[:2][::-1])
        
        # Convert thermal to single channel if needed
        if len(thermal_image.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2GRAY)
        else:
            thermal_gray = thermal_image
        
        # Create 4-channel image (RGB + Thermal)
        if len(rgb_image.shape) == 3:
            combined = np.dstack([rgb_image, thermal_gray])
        else:
            # If RGB is already normalized, handle differently
            combined = np.concatenate([rgb_image, thermal_gray[..., np.newaxis]], axis=-1)
        
        return combined 