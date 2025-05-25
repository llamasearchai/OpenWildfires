"""
OpenWildfires Command Line Interface

A comprehensive CLI for managing the OpenWildfires drone wildfire detection platform.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
import click
import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from openfire.config import get_settings
from openfire.detection import FireDetector, SmokeDetector, EnsembleDetector
from openfire.drone import DroneController, FleetManager
from openfire.ai import OpenAIAnalyzer
from openfire.alerts import AlertSystem

console = Console()
logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version="1.0.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
def cli(verbose: bool, config: Optional[str]):
    """OpenWildfires: Advanced AI-Powered Drone Wildfire Detection Platform"""
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        )
    
    if config:
        # Load custom configuration
        pass
    
    console.print(Panel.fit(
        "[bold red]OpenWildfires[/bold red]\n"
        "[dim]Advanced AI-Powered Drone Wildfire Detection Platform[/dim]\n"
        "[dim]Author: Nik Jois <nikjois@llamasearch.ai>[/dim]",
        border_style="red"
    ))


@cli.group()
def detect():
    """Fire and smoke detection commands"""
    pass


@detect.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--model", "-m", default="yolov8-fire-v2", help="Detection model to use")
@click.option("--confidence", "-c", default=0.5, help="Confidence threshold")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--visualize", "-v", is_flag=True, help="Save visualization")
def image(image_path: str, model: str, confidence: float, output: Optional[str], visualize: bool):
    """Detect fire and smoke in an image"""
    async def run_detection():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading detection model...", total=None)
            
            # Initialize detector
            detector = EnsembleDetector()
            await detector.load_models()
            
            progress.update(task, description="Running detection...")
            
            # Run detection
            result = await detector.detect(image_path)
            
            progress.update(task, description="Processing results...")
            
            # Display results
            table = Table(title="Detection Results")
            table.add_column("Class", style="cyan")
            table.add_column("Confidence", style="magenta")
            table.add_column("Bounding Box", style="green")
            
            for detection in result.detections:
                table.add_row(
                    detection.class_name,
                    f"{detection.confidence:.3f}",
                    f"{detection.bbox}"
                )
            
            console.print(table)
            
            # Summary
            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f"Fire detected: {'Yes' if result.has_fire() else 'No'}")
            console.print(f"Smoke detected: {'Yes' if result.has_smoke() else 'No'}")
            console.print(f"Max confidence: {result.max_confidence():.3f}")
            
            if output:
                # Save results
                import json
                with open(output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                console.print(f"Results saved to {output}")
    
    asyncio.run(run_detection())


@detect.command()
@click.option("--drone-id", "-d", required=True, help="Drone ID to stream from")
@click.option("--duration", "-t", default=60, help="Stream duration in seconds")
@click.option("--confidence", "-c", default=0.5, help="Confidence threshold")
def stream(drone_id: str, duration: int, confidence: float):
    """Real-time detection from drone stream"""
    async def run_stream_detection():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Connecting to drone...", total=None)
            
            # Initialize components
            drone = DroneController(drone_id=drone_id)
            detector = EnsembleDetector()
            
            await drone.connect()
            await detector.load_models()
            
            progress.update(task, description=f"Streaming detection for {duration}s...")
            
            detection_count = 0
            fire_count = 0
            smoke_count = 0
            
            async for frame in drone.camera_stream():
                result = await detector.detect(frame)
                
                if result.detections:
                    detection_count += 1
                    if result.has_fire():
                        fire_count += 1
                    if result.has_smoke():
                        smoke_count += 1
                    
                    console.print(f"[yellow]Detection #{detection_count}:[/yellow] "
                                f"Fire: {result.has_fire()}, "
                                f"Smoke: {result.has_smoke()}, "
                                f"Confidence: {result.max_confidence():.3f}")
                
                # Break after duration (simplified)
                if detection_count >= duration:
                    break
            
            console.print(f"\n[bold]Stream Summary:[/bold]")
            console.print(f"Total detections: {detection_count}")
            console.print(f"Fire detections: {fire_count}")
            console.print(f"Smoke detections: {smoke_count}")
    
    asyncio.run(run_stream_detection())


@cli.group()
def drone():
    """Drone control and management commands"""
    pass


@drone.command()
def list():
    """List all connected drones"""
    async def list_drones():
        fleet = FleetManager()
        drones = await fleet.get_all_drones()
        
        table = Table(title="Connected Drones")
        table.add_column("ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Battery", style="yellow")
        table.add_column("Location", style="blue")
        table.add_column("Mode", style="magenta")
        
        for drone_id, status in drones.items():
            location = f"{status.gps_location.lat:.6f}, {status.gps_location.lon:.6f}" if status.gps_location else "Unknown"
            table.add_row(
                drone_id,
                "Connected" if status.is_connected else "Disconnected",
                f"{status.battery_level:.1f}%",
                location,
                status.mode
            )
        
        console.print(table)
    
    asyncio.run(list_drones())


@drone.command()
@click.argument("drone_id")
@click.option("--altitude", "-a", default=50.0, help="Takeoff altitude in meters")
def takeoff(drone_id: str, altitude: float):
    """Take off a specific drone"""
    async def takeoff_drone():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Connecting to drone...", total=None)
            
            drone = DroneController(drone_id=drone_id)
            await drone.connect()
            
            progress.update(task, description="Arming drone...")
            await drone.arm()
            
            progress.update(task, description=f"Taking off to {altitude}m...")
            success = await drone.takeoff(altitude)
            
            if success:
                console.print(f"[green]✓[/green] Drone {drone_id} successfully took off to {altitude}m")
            else:
                console.print(f"[red]✗[/red] Failed to take off drone {drone_id}")
    
    asyncio.run(takeoff_drone())


@drone.command()
@click.argument("drone_id")
def land(drone_id: str):
    """Land a specific drone"""
    async def land_drone():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Landing drone...", total=None)
            
            drone = DroneController(drone_id=drone_id)
            await drone.connect()
            
            success = await drone.land()
            
            if success:
                console.print(f"[green]✓[/green] Drone {drone_id} successfully landed")
            else:
                console.print(f"[red]✗[/red] Failed to land drone {drone_id}")
    
    asyncio.run(land_drone())


@drone.command()
@click.argument("drone_id")
@click.argument("waypoints_file", type=click.Path(exists=True))
@click.option("--altitude", "-a", default=50.0, help="Flight altitude")
def mission(drone_id: str, waypoints_file: str, altitude: float):
    """Start an autonomous mission"""
    async def start_mission():
        import json
        
        with open(waypoints_file, 'r') as f:
            waypoints = json.load(f)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting mission...", total=None)
            
            drone = DroneController(drone_id=drone_id)
            await drone.connect()
            await drone.arm()
            await drone.takeoff(altitude)
            
            progress.update(task, description="Executing waypoint mission...")
            await drone.start_patrol_mission(waypoints, altitude)
            
            console.print(f"[green]✓[/green] Mission started for drone {drone_id}")
    
    asyncio.run(start_mission())


@cli.group()
def ai():
    """AI analysis and optimization commands"""
    pass


@ai.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--weather", "-w", type=click.Path(exists=True), help="Weather data JSON file")
@click.option("--terrain", "-t", type=click.Path(exists=True), help="Terrain data JSON file")
def analyze(image_path: str, weather: Optional[str], terrain: Optional[str]):
    """Analyze scene using OpenAI"""
    async def run_analysis():
        import json
        
        weather_data = None
        terrain_data = None
        
        if weather:
            with open(weather, 'r') as f:
                weather_data = json.load(f)
        
        if terrain:
            with open(terrain, 'r') as f:
                terrain_data = json.load(f)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing scene with AI...", total=None)
            
            analyzer = OpenAIAnalyzer()
            
            import cv2
            image = cv2.imread(image_path)
            
            analysis = await analyzer.analyze_scene(
                image, 
                weather_data=weather_data,
                terrain_data=terrain_data
            )
            
            # Display results
            console.print(Panel(
                f"[bold]Scene Analysis[/bold]\n\n"
                f"Description: {analysis.description}\n"
                f"Fire Detected: {'Yes' if analysis.fire_detected else 'No'}\n"
                f"Smoke Detected: {'Yes' if analysis.smoke_detected else 'No'}\n"
                f"Risk Level: {analysis.risk_level.upper()}\n"
                f"Confidence: {analysis.confidence:.3f}\n\n"
                f"[bold]Recommendations:[/bold]\n" + 
                "\n".join(f"• {rec}" for rec in analysis.recommendations),
                border_style="blue"
            ))
    
    asyncio.run(run_analysis())


@cli.group()
def alerts():
    """Alert system management commands"""
    pass


@alerts.command()
@click.option("--active-only", is_flag=True, help="Show only active alerts")
def list(active_only: bool):
    """List all alerts"""
    async def list_alerts():
        alert_system = AlertSystem()
        alerts = await alert_system.get_alerts(active_only=active_only)
        
        table = Table(title="Alerts")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="red")
        table.add_column("Priority", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Created", style="blue")
        
        for alert in alerts:
            table.add_row(
                alert.get("id", ""),
                alert.get("type", ""),
                alert.get("priority", ""),
                alert.get("status", ""),
                alert.get("created_at", "")
            )
        
        console.print(table)
    
    asyncio.run(list_alerts())


@alerts.command()
@click.argument("alert_type")
@click.argument("message")
@click.option("--priority", "-p", default="medium", help="Alert priority")
@click.option("--location", "-l", help="Location as 'lat,lon'")
def send(alert_type: str, message: str, priority: str, location: Optional[str]):
    """Send a test alert"""
    async def send_alert():
        alert_system = AlertSystem()
        
        location_coords = None
        if location:
            lat, lon = map(float, location.split(','))
            location_coords = (lat, lon)
        
        await alert_system.send_alert(
            alert_type=alert_type,
            message=message,
            priority=priority,
            location=location_coords
        )
        
        console.print(f"[green]✓[/green] Alert sent: {alert_type}")
    
    asyncio.run(send_alert())


@cli.group()
def system():
    """System management commands"""
    pass


@system.command()
def status():
    """Show system status"""
    async def show_status():
        settings = get_settings()
        
        # Check component status
        status_table = Table(title="System Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="blue")
        
        # Database
        try:
            # Simple database check
            status_table.add_row("Database", "✓ Connected", settings.database.url)
        except Exception:
            status_table.add_row("Database", "✗ Error", "Connection failed")
        
        # Redis
        try:
            status_table.add_row("Redis", "✓ Connected", settings.redis.url)
        except Exception:
            status_table.add_row("Redis", "✗ Error", "Connection failed")
        
        # OpenAI
        if settings.openai.api_key:
            status_table.add_row("OpenAI", "✓ Configured", settings.openai.model)
        else:
            status_table.add_row("OpenAI", "✗ Not configured", "API key missing")
        
        console.print(status_table)
    
    asyncio.run(show_status())


@system.command()
def init():
    """Initialize the system"""
    async def initialize_system():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing system...", total=None)
            
            # Create directories
            progress.update(task, description="Creating directories...")
            Path("models").mkdir(exist_ok=True)
            Path("data").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            
            # Download models
            progress.update(task, description="Downloading detection models...")
            # Model download logic would go here
            
            progress.update(task, description="Setting up database...")
            # Database initialization logic would go here
            
            console.print("[green]✓[/green] System initialized successfully")
    
    asyncio.run(initialize_system())


def main():
    """Main CLI entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main() 