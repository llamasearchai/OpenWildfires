"""
Advanced OpenAI-powered scene analysis and risk assessment.
"""

import asyncio
import base64
import io
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import openai
import structlog

from openfire.config import get_settings

logger = structlog.get_logger(__name__)


class SceneAnalysis:
    """Represents the result of AI scene analysis."""
    
    def __init__(
        self,
        description: str,
        fire_detected: bool,
        smoke_detected: bool,
        risk_level: str,
        confidence: float,
        recommendations: List[str],
        weather_impact: Optional[str] = None,
        terrain_analysis: Optional[str] = None,
        spread_prediction: Optional[str] = None
    ):
        self.description = description
        self.fire_detected = fire_detected
        self.smoke_detected = smoke_detected
        self.risk_level = risk_level  # "low", "medium", "high", "critical"
        self.confidence = confidence
        self.recommendations = recommendations
        self.weather_impact = weather_impact
        self.terrain_analysis = terrain_analysis
        self.spread_prediction = spread_prediction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "description": self.description,
            "fire_detected": self.fire_detected,
            "smoke_detected": self.smoke_detected,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "weather_impact": self.weather_impact,
            "terrain_analysis": self.terrain_analysis,
            "spread_prediction": self.spread_prediction
        }


class OpenAIAnalyzer:
    """Advanced AI analyzer using OpenAI's vision and language models."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = openai.AsyncOpenAI(api_key=self.settings.openai.api_key)
        
    async def analyze_scene(
        self,
        image: np.ndarray,
        context: str = "Wildfire detection and monitoring",
        include_risk_assessment: bool = True,
        weather_data: Optional[Dict[str, Any]] = None,
        terrain_data: Optional[Dict[str, Any]] = None
    ) -> SceneAnalysis:
        """Perform comprehensive scene analysis using OpenAI Vision."""
        try:
            # Convert image to base64
            image_b64 = self._encode_image(image)
            
            # Create comprehensive prompt
            prompt = self._create_analysis_prompt(
                context, include_risk_assessment, weather_data, terrain_data
            )
            
            # Call OpenAI Vision API
            response = await self.client.chat.completions.create(
                model=self.settings.openai.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.settings.openai.max_tokens,
                temperature=self.settings.openai.temperature
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            analysis = self._parse_analysis_response(analysis_text)
            
            logger.info(
                "Scene analysis completed",
                fire_detected=analysis.fire_detected,
                smoke_detected=analysis.smoke_detected,
                risk_level=analysis.risk_level,
                confidence=analysis.confidence
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return self._create_fallback_analysis()
    
    async def assess_fire_behavior(
        self,
        image: np.ndarray,
        weather_data: Dict[str, Any],
        terrain_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Assess fire behavior and predict spread patterns."""
        try:
            image_b64 = self._encode_image(image)
            
            prompt = f"""
            As a wildfire behavior expert, analyze this image and provide detailed fire behavior assessment.
            
            Weather conditions:
            - Wind speed: {weather_data.get('wind_speed', 'unknown')} km/h
            - Wind direction: {weather_data.get('wind_direction', 'unknown')}
            - Temperature: {weather_data.get('temperature', 'unknown')}°C
            - Humidity: {weather_data.get('humidity', 'unknown')}%
            - Precipitation: {weather_data.get('precipitation', 'unknown')} mm
            
            Terrain information:
            - Elevation: {terrain_data.get('elevation', 'unknown')} m
            - Slope: {terrain_data.get('slope', 'unknown')}°
            - Aspect: {terrain_data.get('aspect', 'unknown')}
            - Vegetation type: {terrain_data.get('vegetation', 'unknown')}
            - Fuel load: {terrain_data.get('fuel_load', 'unknown')}
            
            Provide analysis in JSON format with:
            {{
                "fire_intensity": "low/medium/high/extreme",
                "spread_rate": "slow/moderate/fast/extreme",
                "spread_direction": "direction in degrees",
                "flame_height": "estimated height in meters",
                "spotting_potential": "low/medium/high",
                "suppression_difficulty": "easy/moderate/difficult/extreme",
                "evacuation_priority": "low/medium/high/immediate",
                "resource_requirements": ["list of required resources"],
                "tactical_recommendations": ["list of tactical recommendations"],
                "time_to_containment": "estimated hours",
                "confidence": 0.0-1.0
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.settings.openai.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            # Parse JSON response
            import json
            behavior_analysis = json.loads(response.choices[0].message.content)
            
            logger.info("Fire behavior analysis completed", **behavior_analysis)
            return behavior_analysis
            
        except Exception as e:
            logger.error(f"Fire behavior analysis failed: {e}")
            return self._create_fallback_behavior_analysis()
    
    async def generate_incident_report(
        self,
        detections: List[Dict[str, Any]],
        drone_data: Dict[str, Any],
        weather_data: Dict[str, Any],
        analysis_results: List[SceneAnalysis]
    ) -> str:
        """Generate a comprehensive incident report."""
        try:
            prompt = f"""
            Generate a professional wildfire incident report based on the following data:
            
            DETECTION SUMMARY:
            {self._format_detections_for_report(detections)}
            
            DRONE TELEMETRY:
            - Location: {drone_data.get('location', 'unknown')}
            - Altitude: {drone_data.get('altitude', 'unknown')} m
            - Time: {drone_data.get('timestamp', 'unknown')}
            - Battery: {drone_data.get('battery', 'unknown')}%
            
            WEATHER CONDITIONS:
            {self._format_weather_for_report(weather_data)}
            
            AI ANALYSIS RESULTS:
            {self._format_analysis_for_report(analysis_results)}
            
            Generate a professional incident report including:
            1. Executive Summary
            2. Incident Details
            3. Risk Assessment
            4. Recommended Actions
            5. Resource Requirements
            6. Timeline and Priorities
            
            Format as a structured report suitable for emergency responders.
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.2
            )
            
            report = response.choices[0].message.content
            logger.info("Incident report generated")
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return "Error generating incident report. Please contact system administrator."
    
    async def optimize_drone_deployment(
        self,
        fire_locations: List[Tuple[float, float]],
        available_drones: List[Dict[str, Any]],
        weather_data: Dict[str, Any],
        terrain_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize drone deployment strategy using AI."""
        try:
            prompt = f"""
            As a drone deployment strategist for wildfire response, optimize the deployment of available drones.
            
            FIRE LOCATIONS:
            {fire_locations}
            
            AVAILABLE DRONES:
            {available_drones}
            
            WEATHER CONDITIONS:
            {weather_data}
            
            TERRAIN DATA:
            {terrain_data}
            
            Provide optimal deployment strategy in JSON format:
            {{
                "deployment_plan": [
                    {{
                        "drone_id": "string",
                        "assigned_location": [lat, lon],
                        "priority": "high/medium/low",
                        "mission_type": "surveillance/suppression/evacuation",
                        "flight_altitude": "meters",
                        "estimated_flight_time": "minutes",
                        "battery_requirements": "percentage"
                    }}
                ],
                "coverage_analysis": {{
                    "total_coverage_area": "square_km",
                    "overlap_percentage": "percentage",
                    "blind_spots": ["list of coordinates"]
                }},
                "risk_mitigation": ["list of risk mitigation strategies"],
                "backup_plan": "description of backup deployment",
                "confidence": 0.0-1.0
            }}
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            import json
            deployment_plan = json.loads(response.choices[0].message.content)
            
            logger.info("Drone deployment optimization completed")
            return deployment_plan
            
        except Exception as e:
            logger.error(f"Deployment optimization failed: {e}")
            return self._create_fallback_deployment_plan()
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 string."""
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Convert to JPEG
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        
        # Encode to base64
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        return image_b64
    
    def _create_analysis_prompt(
        self,
        context: str,
        include_risk_assessment: bool,
        weather_data: Optional[Dict[str, Any]],
        terrain_data: Optional[Dict[str, Any]]
    ) -> str:
        """Create comprehensive analysis prompt."""
        prompt = f"""
        You are an expert wildfire detection and analysis system. Analyze this image for fire and smoke detection.
        
        Context: {context}
        
        Provide detailed analysis including:
        1. Scene description
        2. Fire detection (yes/no and confidence)
        3. Smoke detection (yes/no and confidence)
        4. Risk level assessment (low/medium/high/critical)
        5. Specific recommendations for response
        """
        
        if weather_data:
            prompt += f"\n\nWeather conditions to consider:\n{weather_data}"
        
        if terrain_data:
            prompt += f"\n\nTerrain information:\n{terrain_data}"
        
        if include_risk_assessment:
            prompt += """
            
            Include comprehensive risk assessment covering:
            - Immediate fire spread risk
            - Threat to structures/infrastructure
            - Evacuation recommendations
            - Resource deployment priorities
            - Weather impact on fire behavior
            - Terrain influence on fire spread
            """
        
        prompt += """
        
        Format your response as structured text that can be parsed, including:
        - DESCRIPTION: [detailed scene description]
        - FIRE_DETECTED: [true/false]
        - SMOKE_DETECTED: [true/false]
        - RISK_LEVEL: [low/medium/high/critical]
        - CONFIDENCE: [0.0-1.0]
        - RECOMMENDATIONS: [numbered list of specific actions]
        - WEATHER_IMPACT: [how weather affects the situation]
        - TERRAIN_ANALYSIS: [how terrain influences fire behavior]
        - SPREAD_PREDICTION: [predicted fire spread pattern and timeline]
        """
        
        return prompt
    
    def _parse_analysis_response(self, response_text: str) -> SceneAnalysis:
        """Parse the AI response into structured analysis."""
        try:
            # Simple parsing - in production, you'd want more robust parsing
            lines = response_text.split('\n')
            
            description = ""
            fire_detected = False
            smoke_detected = False
            risk_level = "low"
            confidence = 0.5
            recommendations = []
            weather_impact = None
            terrain_analysis = None
            spread_prediction = None
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("FIRE_DETECTED:"):
                    fire_detected = "true" in line.lower()
                elif line.startswith("SMOKE_DETECTED:"):
                    smoke_detected = "true" in line.lower()
                elif line.startswith("RISK_LEVEL:"):
                    risk_level = line.replace("RISK_LEVEL:", "").strip().lower()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.replace("CONFIDENCE:", "").strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith("RECOMMENDATIONS:"):
                    current_section = "recommendations"
                elif line.startswith("WEATHER_IMPACT:"):
                    weather_impact = line.replace("WEATHER_IMPACT:", "").strip()
                elif line.startswith("TERRAIN_ANALYSIS:"):
                    terrain_analysis = line.replace("TERRAIN_ANALYSIS:", "").strip()
                elif line.startswith("SPREAD_PREDICTION:"):
                    spread_prediction = line.replace("SPREAD_PREDICTION:", "").strip()
                elif current_section == "recommendations" and line.startswith(("-", "•", "1.", "2.", "3.", "4.", "5.")):
                    recommendations.append(line.lstrip("-•123456789. "))
            
            return SceneAnalysis(
                description=description,
                fire_detected=fire_detected,
                smoke_detected=smoke_detected,
                risk_level=risk_level,
                confidence=confidence,
                recommendations=recommendations,
                weather_impact=weather_impact,
                terrain_analysis=terrain_analysis,
                spread_prediction=spread_prediction
            )
            
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return self._create_fallback_analysis()
    
    def _create_fallback_analysis(self) -> SceneAnalysis:
        """Create fallback analysis when AI analysis fails."""
        return SceneAnalysis(
            description="Analysis unavailable due to system error",
            fire_detected=False,
            smoke_detected=False,
            risk_level="unknown",
            confidence=0.0,
            recommendations=["Manual inspection required", "Contact system administrator"],
            weather_impact="Unable to assess",
            terrain_analysis="Unable to assess",
            spread_prediction="Unable to predict"
        )
    
    def _create_fallback_behavior_analysis(self) -> Dict[str, Any]:
        """Create fallback fire behavior analysis."""
        return {
            "fire_intensity": "unknown",
            "spread_rate": "unknown",
            "spread_direction": "unknown",
            "flame_height": "unknown",
            "spotting_potential": "unknown",
            "suppression_difficulty": "unknown",
            "evacuation_priority": "medium",
            "resource_requirements": ["Manual assessment required"],
            "tactical_recommendations": ["Conduct ground reconnaissance"],
            "time_to_containment": "unknown",
            "confidence": 0.0
        }
    
    def _create_fallback_deployment_plan(self) -> Dict[str, Any]:
        """Create fallback deployment plan."""
        return {
            "deployment_plan": [],
            "coverage_analysis": {
                "total_coverage_area": "unknown",
                "overlap_percentage": "unknown",
                "blind_spots": []
            },
            "risk_mitigation": ["Manual deployment planning required"],
            "backup_plan": "Revert to manual deployment procedures",
            "confidence": 0.0
        }
    
    def _format_detections_for_report(self, detections: List[Dict[str, Any]]) -> str:
        """Format detection data for report."""
        if not detections:
            return "No detections recorded"
        
        formatted = []
        for i, detection in enumerate(detections, 1):
            formatted.append(f"{i}. {detection.get('class_name', 'unknown')} - "
                           f"Confidence: {detection.get('confidence', 0):.2f} - "
                           f"Location: {detection.get('bbox', 'unknown')}")
        
        return "\n".join(formatted)
    
    def _format_weather_for_report(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data for report."""
        return f"""
        - Temperature: {weather_data.get('temperature', 'unknown')}°C
        - Wind Speed: {weather_data.get('wind_speed', 'unknown')} km/h
        - Wind Direction: {weather_data.get('wind_direction', 'unknown')}
        - Humidity: {weather_data.get('humidity', 'unknown')}%
        - Precipitation: {weather_data.get('precipitation', 'unknown')} mm
        """
    
    def _format_analysis_for_report(self, analyses: List[SceneAnalysis]) -> str:
        """Format analysis results for report."""
        if not analyses:
            return "No AI analysis available"
        
        formatted = []
        for i, analysis in enumerate(analyses, 1):
            formatted.append(f"""
            Analysis {i}:
            - Fire Detected: {analysis.fire_detected}
            - Smoke Detected: {analysis.smoke_detected}
            - Risk Level: {analysis.risk_level}
            - Confidence: {analysis.confidence:.2f}
            - Description: {analysis.description}
            """)
        
        return "\n".join(formatted) 