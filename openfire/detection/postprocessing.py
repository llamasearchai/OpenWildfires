"""
Advanced postprocessing for detection results.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import cv2
import structlog

logger = structlog.get_logger(__name__)


class DetectionPostprocessor:
    """Advanced postprocessing for detection results."""
    
    def __init__(
        self,
        nms_threshold: float = 0.4,
        confidence_threshold: float = 0.5,
        min_area: int = 100,
        max_area: int = 50000
    ):
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self.max_area = max_area
    
    def process(self, detections: List['Detection']) -> List['Detection']:
        """Process detection results through postprocessing pipeline."""
        if not detections:
            return []
        
        # Filter by confidence
        filtered_detections = self._filter_by_confidence(detections)
        
        # Filter by area
        filtered_detections = self._filter_by_area(filtered_detections)
        
        # Apply Non-Maximum Suppression
        nms_detections = self._apply_nms(filtered_detections)
        
        # Apply temporal filtering if applicable
        temporal_detections = self._apply_temporal_filtering(nms_detections)
        
        # Sort by confidence
        final_detections = sorted(temporal_detections, key=lambda x: x.confidence, reverse=True)
        
        logger.info(
            f"Postprocessing complete: {len(detections)} -> {len(final_detections)} detections",
            original_count=len(detections),
            final_count=len(final_detections)
        )
        
        return final_detections
    
    def _filter_by_confidence(self, detections: List['Detection']) -> List['Detection']:
        """Filter detections by confidence threshold."""
        return [d for d in detections if d.confidence >= self.confidence_threshold]
    
    def _filter_by_area(self, detections: List['Detection']) -> List['Detection']:
        """Filter detections by bounding box area."""
        filtered = []
        for detection in detections:
            area = detection.area
            if self.min_area <= area <= self.max_area:
                filtered.append(detection)
            else:
                logger.debug(
                    f"Filtered detection by area: {area}",
                    min_area=self.min_area,
                    max_area=self.max_area
                )
        return filtered
    
    def _apply_nms(self, detections: List['Detection']) -> List['Detection']:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if len(detections) <= 1:
            return detections
        
        # Group detections by class
        class_groups = {}
        for detection in detections:
            class_name = detection.class_name
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(detection)
        
        # Apply NMS per class
        final_detections = []
        for class_name, class_detections in class_groups.items():
            nms_detections = self._nms_per_class(class_detections)
            final_detections.extend(nms_detections)
        
        return final_detections
    
    def _nms_per_class(self, detections: List['Detection']) -> List['Detection']:
        """Apply NMS for a single class."""
        if len(detections) <= 1:
            return detections
        
        # Convert to format suitable for OpenCV NMS
        boxes = []
        scores = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, w, h)
            scores.append(detection.confidence)
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Return filtered detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []
    
    def _apply_temporal_filtering(self, detections: List['Detection']) -> List['Detection']:
        """Apply temporal filtering to reduce false positives."""
        # This would be implemented with a detection history buffer
        # For now, return detections as-is
        return detections


class FirePostprocessor(DetectionPostprocessor):
    """Specialized postprocessor for fire detections."""
    
    def __init__(
        self,
        nms_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        min_area: int = 200,
        max_area: int = 100000
    ):
        super().__init__(nms_threshold, confidence_threshold, min_area, max_area)
        self.fire_color_validator = FireColorValidator()
    
    def process(self, detections: List['Detection']) -> List['Detection']:
        """Process fire detections with additional validation."""
        # Apply base postprocessing
        processed_detections = super().process(detections)
        
        # Apply fire-specific validation
        validated_detections = self._validate_fire_detections(processed_detections)
        
        return validated_detections
    
    def _validate_fire_detections(self, detections: List['Detection']) -> List['Detection']:
        """Validate fire detections using color and shape analysis."""
        validated = []
        
        for detection in detections:
            if detection.class_name == "fire":
                # Additional validation for fire detections
                if self._is_valid_fire_detection(detection):
                    validated.append(detection)
                else:
                    logger.debug(f"Fire detection failed validation: {detection.bbox}")
            else:
                validated.append(detection)
        
        return validated
    
    def _is_valid_fire_detection(self, detection: 'Detection') -> bool:
        """Validate a fire detection using additional criteria."""
        # Check aspect ratio (fires are usually not extremely elongated)
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else float('inf')
        
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            return False
        
        # Additional validation could include:
        # - Color analysis within the bounding box
        # - Texture analysis
        # - Motion analysis (if temporal data available)
        
        return True


class SmokePostprocessor(DetectionPostprocessor):
    """Specialized postprocessor for smoke detections."""
    
    def __init__(
        self,
        nms_threshold: float = 0.5,
        confidence_threshold: float = 0.4,
        min_area: int = 500,
        max_area: int = 200000
    ):
        super().__init__(nms_threshold, confidence_threshold, min_area, max_area)
    
    def process(self, detections: List['Detection']) -> List['Detection']:
        """Process smoke detections with additional validation."""
        # Apply base postprocessing
        processed_detections = super().process(detections)
        
        # Apply smoke-specific validation
        validated_detections = self._validate_smoke_detections(processed_detections)
        
        return validated_detections
    
    def _validate_smoke_detections(self, detections: List['Detection']) -> List['Detection']:
        """Validate smoke detections using shape and texture analysis."""
        validated = []
        
        for detection in detections:
            if detection.class_name == "smoke":
                if self._is_valid_smoke_detection(detection):
                    validated.append(detection)
                else:
                    logger.debug(f"Smoke detection failed validation: {detection.bbox}")
            else:
                validated.append(detection)
        
        return validated
    
    def _is_valid_smoke_detection(self, detection: 'Detection') -> bool:
        """Validate a smoke detection using additional criteria."""
        # Smoke typically has larger, more irregular shapes
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Check if the detection is too small for smoke
        if area < 1000:
            return False
        
        # Smoke usually has a more vertical orientation
        aspect_ratio = height / width if width > 0 else float('inf')
        if aspect_ratio < 0.5:  # Too wide for typical smoke
            return False
        
        return True


class FireColorValidator:
    """Validates fire detections based on color analysis."""
    
    def __init__(self):
        # Define fire color ranges in HSV
        self.fire_ranges = [
            # Red range
            ((0, 50, 50), (10, 255, 255)),
            ((170, 50, 50), (180, 255, 255)),
            # Orange range
            ((10, 50, 50), (25, 255, 255)),
            # Yellow range
            ((25, 50, 50), (35, 255, 255))
        ]
    
    def validate(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Validate fire detection based on color content."""
        x1, y1, x2, y2 = bbox
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Check for fire colors
        fire_pixel_count = 0
        total_pixels = roi.shape[0] * roi.shape[1]
        
        for lower, upper in self.fire_ranges:
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            fire_pixel_count += np.sum(mask > 0)
        
        # Calculate fire color percentage
        fire_percentage = fire_pixel_count / total_pixels
        
        # Require at least 20% fire-colored pixels
        return fire_percentage >= 0.2


class TemporalFilter:
    """Temporal filtering to reduce false positives across frames."""
    
    def __init__(
        self,
        history_size: int = 10,
        confirmation_threshold: int = 3
    ):
        self.history_size = history_size
        self.confirmation_threshold = confirmation_threshold
        self.detection_history = []
    
    def filter(self, detections: List['Detection']) -> List['Detection']:
        """Filter detections based on temporal consistency."""
        # Add current detections to history
        self.detection_history.append(detections)
        
        # Keep only recent history
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # If we don't have enough history, return current detections
        if len(self.detection_history) < self.confirmation_threshold:
            return detections
        
        # Find detections that appear consistently
        confirmed_detections = []
        
        for detection in detections:
            confirmation_count = self._count_confirmations(detection)
            if confirmation_count >= self.confirmation_threshold:
                confirmed_detections.append(detection)
        
        return confirmed_detections
    
    def _count_confirmations(self, detection: 'Detection') -> int:
        """Count how many times a similar detection appeared in history."""
        count = 0
        
        for historical_detections in self.detection_history:
            for hist_detection in historical_detections:
                if self._are_similar_detections(detection, hist_detection):
                    count += 1
                    break
        
        return count
    
    def _are_similar_detections(
        self,
        det1: 'Detection',
        det2: 'Detection'
    ) -> bool:
        """Check if two detections are similar (same object)."""
        # Check class match
        if det1.class_name != det2.class_name:
            return False
        
        # Check spatial overlap
        iou = self._calculate_iou(det1.bbox, det2.bbox)
        return iou > 0.3
    
    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
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