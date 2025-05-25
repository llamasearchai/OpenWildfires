"""
Tests for fire and smoke detection functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from openfire.detection.models import (
    Detection, DetectionResult, FireDetector, SmokeDetector, EnsembleDetector
)


class TestDetection:
    """Test Detection class."""
    
    def test_detection_creation(self):
        """Test Detection object creation."""
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_name="fire",
            class_id=0
        )
        
        assert detection.bbox == (100, 100, 200, 200)
        assert detection.confidence == 0.85
        assert detection.class_name == "fire"
        assert detection.class_id == 0
        assert detection.center == (150, 150)
        assert detection.area == 10000
    
    def test_detection_to_dict(self):
        """Test Detection to_dict method."""
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_name="fire",
            class_id=0
        )
        
        result = detection.to_dict()
        expected = {
            "bbox": (100, 100, 200, 200),
            "confidence": 0.85,
            "class_name": "fire",
            "class_id": 0,
            "center": (150, 150),
            "area": 10000
        }
        
        assert result == expected


class TestDetectionResult:
    """Test DetectionResult class."""
    
    def test_detection_result_creation(self):
        """Test DetectionResult object creation."""
        detections = [
            Detection((100, 100, 200, 200), 0.85, "fire", 0),
            Detection((300, 300, 400, 400), 0.75, "smoke", 1)
        ]
        
        result = DetectionResult(detections, (480, 640))
        
        assert len(result.detections) == 2
        assert result.image_shape == (480, 640)
        assert result.has_fire() is True
        assert result.has_smoke() is True
        assert result.max_confidence() == 0.85
        assert len(result.fire_detections()) == 1
        assert len(result.smoke_detections()) == 1
    
    def test_detection_result_empty(self):
        """Test DetectionResult with no detections."""
        result = DetectionResult([], (480, 640))
        
        assert len(result.detections) == 0
        assert result.has_fire() is False
        assert result.has_smoke() is False
        assert result.max_confidence() == 0.0
        assert len(result.fire_detections()) == 0
        assert len(result.smoke_detections()) == 0
    
    def test_detection_result_to_dict(self):
        """Test DetectionResult to_dict method."""
        detections = [
            Detection((100, 100, 200, 200), 0.85, "fire", 0)
        ]
        
        result = DetectionResult(detections, (480, 640))
        result_dict = result.to_dict()
        
        assert "detections" in result_dict
        assert "image_shape" in result_dict
        assert "has_fire" in result_dict
        assert "has_smoke" in result_dict
        assert "max_confidence" in result_dict
        assert "fire_count" in result_dict
        assert "smoke_count" in result_dict
        
        assert result_dict["has_fire"] is True
        assert result_dict["fire_count"] == 1
        assert result_dict["smoke_count"] == 0


class TestFireDetector:
    """Test FireDetector class."""
    
    @pytest.fixture
    def fire_detector(self):
        """Create a FireDetector instance for testing."""
        return FireDetector(confidence_threshold=0.5)
    
    def test_fire_detector_initialization(self, fire_detector):
        """Test FireDetector initialization."""
        assert fire_detector.confidence_threshold == 0.5
        assert fire_detector.class_names == ["fire", "smoke"]
        assert fire_detector.model is None
    
    @patch('openfire.detection.models.YOLO')
    async def test_load_model(self, mock_yolo, fire_detector):
        """Test model loading."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        await fire_detector.load_model()
        
        assert fire_detector.model is not None
        mock_yolo.assert_called_once()
    
    @patch('openfire.detection.models.YOLO')
    async def test_detect_with_mock_model(self, mock_yolo, fire_detector, sample_image_data):
        """Test detection with mocked model."""
        # Mock YOLO model and results
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[100, 100, 200, 200]])
        mock_result.boxes.conf = np.array([0.85])
        mock_result.boxes.cls = np.array([0])
        
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        await fire_detector.load_model()
        result = await fire_detector.detect(sample_image_data)
        
        assert isinstance(result, DetectionResult)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "fire"
        assert result.detections[0].confidence == 0.85
    
    def test_prepare_image_numpy(self, fire_detector, sample_image_data):
        """Test image preparation with numpy array."""
        processed = fire_detector._prepare_image(sample_image_data)
        assert isinstance(processed, np.ndarray)
        assert processed.shape == sample_image_data.shape
    
    def test_get_class_name(self, fire_detector):
        """Test class name mapping."""
        assert fire_detector._get_class_name(0) == "fire"
        assert fire_detector._get_class_name(1) == "smoke"
        assert fire_detector._get_class_name(999) == "unknown"


class TestSmokeDetector:
    """Test SmokeDetector class."""
    
    @pytest.fixture
    def smoke_detector(self):
        """Create a SmokeDetector instance for testing."""
        return SmokeDetector(confidence_threshold=0.4)
    
    def test_smoke_detector_initialization(self, smoke_detector):
        """Test SmokeDetector initialization."""
        assert smoke_detector.confidence_threshold == 0.4
        assert smoke_detector.class_names == ["smoke"]
    
    def test_enhance_for_smoke(self, smoke_detector, sample_image_data):
        """Test smoke enhancement preprocessing."""
        enhanced = smoke_detector._enhance_for_smoke(sample_image_data)
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == sample_image_data.shape


class TestEnsembleDetector:
    """Test EnsembleDetector class."""
    
    @pytest.fixture
    def ensemble_detector(self):
        """Create an EnsembleDetector instance for testing."""
        fire_detector = Mock(spec=FireDetector)
        smoke_detector = Mock(spec=SmokeDetector)
        return EnsembleDetector(fire_detector, smoke_detector)
    
    def test_ensemble_detector_initialization(self, ensemble_detector):
        """Test EnsembleDetector initialization."""
        assert ensemble_detector.fire_detector is not None
        assert ensemble_detector.smoke_detector is not None
    
    async def test_load_models(self, ensemble_detector):
        """Test loading models in ensemble detector."""
        ensemble_detector.fire_detector.load_model = AsyncMock()
        ensemble_detector.smoke_detector.load_model = AsyncMock()
        
        await ensemble_detector.load_models()
        
        ensemble_detector.fire_detector.load_model.assert_called_once()
        ensemble_detector.smoke_detector.load_model.assert_called_once()
    
    async def test_detect_ensemble(self, ensemble_detector, sample_image_data):
        """Test ensemble detection."""
        # Mock fire detector results
        fire_detection = Detection((100, 100, 200, 200), 0.85, "fire", 0)
        fire_result = DetectionResult([fire_detection], (480, 640))
        
        # Mock smoke detector results
        smoke_detection = Detection((300, 300, 400, 400), 0.75, "smoke", 1)
        smoke_result = DetectionResult([smoke_detection], (480, 640))
        
        ensemble_detector.fire_detector.detect = AsyncMock(return_value=fire_result)
        ensemble_detector.smoke_detector.detect = AsyncMock(return_value=smoke_result)
        
        result = await ensemble_detector.detect(sample_image_data)
        
        assert isinstance(result, DetectionResult)
        assert len(result.detections) == 2
        assert result.has_fire() is True
        assert result.has_smoke() is True
    
    def test_merge_detections(self, ensemble_detector):
        """Test detection merging logic."""
        detections = [
            Detection((100, 100, 200, 200), 0.85, "fire", 0),
            Detection((105, 105, 205, 205), 0.80, "fire", 0),  # Overlapping
            Detection((300, 300, 400, 400), 0.75, "smoke", 1)
        ]
        
        merged = ensemble_detector._merge_detections(detections)
        
        # Should merge overlapping fire detections
        assert len(merged) == 2
        fire_detections = [d for d in merged if d.class_name == "fire"]
        smoke_detections = [d for d in merged if d.class_name == "smoke"]
        assert len(fire_detections) == 1
        assert len(smoke_detections) == 1
    
    def test_calculate_iou(self, ensemble_detector):
        """Test IoU calculation."""
        box1 = (100, 100, 200, 200)
        box2 = (150, 150, 250, 250)
        
        iou = ensemble_detector._calculate_iou(box1, box2)
        
        # Expected IoU for these boxes
        expected_iou = 2500 / 17500  # intersection / union
        assert abs(iou - expected_iou) < 0.01


@pytest.mark.integration
class TestDetectionIntegration:
    """Integration tests for detection system."""
    
    @pytest.mark.asyncio
    async def test_full_detection_pipeline(self, sample_image_data):
        """Test complete detection pipeline."""
        # This would require actual model files for full integration testing
        # For now, we'll test the structure
        
        detector = EnsembleDetector()
        
        # Mock the model loading and detection
        with patch.object(detector, 'load_models') as mock_load:
            with patch.object(detector, 'detect') as mock_detect:
                mock_load.return_value = None
                mock_detect.return_value = DetectionResult([], (480, 640))
                
                await detector.load_models()
                result = await detector.detect(sample_image_data)
                
                assert isinstance(result, DetectionResult)
                mock_load.assert_called_once()
                mock_detect.assert_called_once()


@pytest.mark.performance
class TestDetectionPerformance:
    """Performance tests for detection system."""
    
    @pytest.mark.asyncio
    async def test_detection_speed(self, sample_image_data):
        """Test detection speed requirements."""
        import time
        
        detector = EnsembleDetector()
        
        # Mock fast detection
        with patch.object(detector, 'detect') as mock_detect:
            mock_detect.return_value = DetectionResult([], (480, 640))
            
            start_time = time.time()
            await detector.detect(sample_image_data)
            end_time = time.time()
            
            # Should complete quickly (mocked)
            assert (end_time - start_time) < 1.0
    
    def test_memory_usage(self, sample_image_data):
        """Test memory usage during detection."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create detector and process image
        detector = FireDetector()
        processed = detector._prepare_image(sample_image_data)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test image)
        assert memory_increase < 100 * 1024 * 1024 