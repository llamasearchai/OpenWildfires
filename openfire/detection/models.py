"""
Advanced fire and smoke detection models.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from PIL import Image
import structlog

from openfire.config import get_settings
from openfire.detection.preprocessing import ImagePreprocessor
from openfire.detection.postprocessing import DetectionPostprocessor

logger = structlog.get_logger(__name__)


class Detection:
    """Represents a single detection result."""
    
    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        class_name: str,
        class_id: int,
        mask: Optional[np.ndarray] = None
    ):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_name = class_name
        self.class_id = class_id
        self.mask = mask
        
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Get the area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "center": self.center,
            "area": self.area
        }


class DetectionResult:
    """Container for detection results."""
    
    def __init__(self, detections: List[Detection], image_shape: Tuple[int, int]):
        self.detections = detections
        self.image_shape = image_shape  # (height, width)
        
    def has_fire(self) -> bool:
        """Check if any fire detections are present."""
        return any(d.class_name == "fire" for d in self.detections)
    
    def has_smoke(self) -> bool:
        """Check if any smoke detections are present."""
        return any(d.class_name == "smoke" for d in self.detections)
    
    def max_confidence(self) -> float:
        """Get the maximum confidence score."""
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)
    
    def fire_detections(self) -> List[Detection]:
        """Get only fire detections."""
        return [d for d in self.detections if d.class_name == "fire"]
    
    def smoke_detections(self) -> List[Detection]:
        """Get only smoke detections."""
        return [d for d in self.detections if d.class_name == "smoke"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "detections": [d.to_dict() for d in self.detections],
            "image_shape": self.image_shape,
            "has_fire": self.has_fire(),
            "has_smoke": self.has_smoke(),
            "max_confidence": self.max_confidence(),
            "fire_count": len(self.fire_detections()),
            "smoke_count": len(self.smoke_detections())
        }


class BaseDetector:
    """Base class for detection models."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cuda"
    ):
        self.settings = get_settings()
        self.model_path = model_path or self.settings.detection.model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = DetectionPostprocessor()
        
    async def load_model(self) -> None:
        """Load the detection model."""
        raise NotImplementedError
    
    async def detect(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> DetectionResult:
        """Detect objects in an image."""
        raise NotImplementedError
    
    def _prepare_image(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> np.ndarray:
        """Prepare image for inference."""
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR and convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.preprocessor.process(image)


class FireDetector(BaseDetector):
    """Advanced fire detection model using YOLOv8."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cuda"
    ):
        super().__init__(model_path, confidence_threshold, device)
        self.class_names = ["fire", "smoke"]
        
    async def load_model(self) -> None:
        """Load the YOLOv8 fire detection model."""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded custom model from {self.model_path}")
            else:
                # Use pre-trained YOLOv8 and fine-tune for fire detection
                self.model = YOLO("yolov8n.pt")
                logger.info("Loaded pre-trained YOLOv8 model")
                
            # Move model to specified device
            if torch.cuda.is_available() and self.device == "cuda":
                self.model.to("cuda")
                logger.info("Model moved to CUDA")
            else:
                self.model.to("cpu")
                logger.info("Model running on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def detect(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> DetectionResult:
        """Detect fire and smoke in an image."""
        if self.model is None:
            await self.load_model()
        
        # Prepare image
        processed_image = self._prepare_image(image)
        original_shape = processed_image.shape[:2]
        
        try:
            # Run inference
            results = self.model(processed_image, conf=self.confidence_threshold)
            
            # Process results
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        if conf >= self.confidence_threshold:
                            # Map class ID to name
                            class_name = self._get_class_name(cls_id)
                            if class_name in ["fire", "smoke"]:
                                detection = Detection(
                                    bbox=tuple(map(int, box)),
                                    confidence=float(conf),
                                    class_name=class_name,
                                    class_id=int(cls_id)
                                )
                                detections.append(detection)
            
            # Apply post-processing
            detections = self.postprocessor.process(detections)
            
            result = DetectionResult(detections, original_shape)
            
            logger.info(
                f"Detection complete: {len(detections)} objects found",
                fire_count=len(result.fire_detections()),
                smoke_count=len(result.smoke_detections()),
                max_confidence=result.max_confidence()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return DetectionResult([], original_shape)
    
    def _get_class_name(self, class_id: int) -> str:
        """Map class ID to class name."""
        # This would be customized based on your trained model
        class_mapping = {
            0: "fire",
            1: "smoke",
            # Add more classes as needed
        }
        return class_mapping.get(class_id, "unknown")


class SmokeDetector(BaseDetector):
    """Specialized smoke detection model."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        device: str = "cuda"
    ):
        super().__init__(model_path, confidence_threshold, device)
        self.class_names = ["smoke"]
        
    async def load_model(self) -> None:
        """Load the smoke detection model."""
        try:
            # Load specialized smoke detection model
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded smoke detection model from {self.model_path}")
            else:
                # Fallback to general model
                self.model = YOLO("yolov8n.pt")
                logger.info("Loaded fallback model for smoke detection")
                
            if torch.cuda.is_available() and self.device == "cuda":
                self.model.to("cuda")
            else:
                self.model.to("cpu")
                
        except Exception as e:
            logger.error(f"Failed to load smoke detection model: {e}")
            raise
    
    async def detect(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> DetectionResult:
        """Detect smoke in an image."""
        if self.model is None:
            await self.load_model()
        
        processed_image = self._prepare_image(image)
        original_shape = processed_image.shape[:2]
        
        try:
            # Enhanced preprocessing for smoke detection
            enhanced_image = self._enhance_for_smoke(processed_image)
            
            # Run inference
            results = self.model(enhanced_image, conf=self.confidence_threshold)
            
            # Process results (similar to FireDetector but smoke-specific)
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        if conf >= self.confidence_threshold:
                            detection = Detection(
                                bbox=tuple(map(int, box)),
                                confidence=float(conf),
                                class_name="smoke",
                                class_id=1
                            )
                            detections.append(detection)
            
            detections = self.postprocessor.process(detections)
            result = DetectionResult(detections, original_shape)
            
            logger.info(
                f"Smoke detection complete: {len(detections)} smoke regions found",
                max_confidence=result.max_confidence()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Smoke detection failed: {e}")
            return DetectionResult([], original_shape)
    
    def _enhance_for_smoke(self, image: np.ndarray) -> np.ndarray:
        """Apply smoke-specific image enhancements."""
        # Convert to grayscale for better smoke detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb


class EnsembleDetector:
    """Ensemble detector combining multiple models for improved accuracy."""
    
    def __init__(
        self,
        fire_detector: Optional[FireDetector] = None,
        smoke_detector: Optional[SmokeDetector] = None
    ):
        self.fire_detector = fire_detector or FireDetector()
        self.smoke_detector = smoke_detector or SmokeDetector()
        
    async def load_models(self) -> None:
        """Load all ensemble models."""
        await asyncio.gather(
            self.fire_detector.load_model(),
            self.smoke_detector.load_model()
        )
    
    async def detect(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> DetectionResult:
        """Run ensemble detection."""
        # Run both detectors in parallel
        fire_result, smoke_result = await asyncio.gather(
            self.fire_detector.detect(image),
            self.smoke_detector.detect(image)
        )
        
        # Combine results
        all_detections = fire_result.detections + smoke_result.detections
        
        # Remove duplicates and apply NMS
        final_detections = self._merge_detections(all_detections)
        
        return DetectionResult(final_detections, fire_result.image_shape)
    
    def _merge_detections(self, detections: List[Detection]) -> List[Detection]:
        """Merge and deduplicate detections from multiple models."""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply Non-Maximum Suppression
        final_detections = []
        for detection in detections:
            # Check if this detection overlaps significantly with existing ones
            is_duplicate = False
            for existing in final_detections:
                if self._calculate_iou(detection.bbox, existing.bbox) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        return final_detections
    
    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0 