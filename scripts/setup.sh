#!/bin/bash

# OpenWildfires Setup Script
# Author: Nik Jois <nikjois@llamasearch.ai>
# License: MIT

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
NODE_VERSION="18"
PROJECT_NAME="OpenWildfires"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! check_command brew; then
            log_error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        
        # Install Python if needed
        if ! check_command python3.11; then
            log_info "Installing Python 3.11..."
            brew install python@3.11
        fi
        
        # Install system dependencies
        brew install postgresql redis gdal proj geos
        
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if check_command apt-get; then
            # Ubuntu/Debian
            sudo apt-get update
            sudo apt-get install -y \
                python3.11 python3.11-venv python3.11-dev \
                postgresql postgresql-contrib redis-server \
                libgdal-dev gdal-bin libproj-dev proj-data proj-bin \
                libgeos-dev libspatialindex-dev \
                build-essential curl git \
                ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
                libswscale-dev libavresample-dev pkg-config
                
        elif check_command yum; then
            # CentOS/RHEL
            sudo yum update -y
            sudo yum install -y \
                python311 python311-devel \
                postgresql postgresql-server redis \
                gdal-devel proj-devel geos-devel \
                gcc gcc-c++ make curl git \
                ffmpeg-devel
        else
            log_error "Unsupported Linux distribution"
            exit 1
        fi
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

install_node_dependencies() {
    log_info "Installing Node.js dependencies..."
    
    if ! check_command node; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install node@18
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Install Node.js via NodeSource
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
        fi
    fi
    
    # Verify Node.js version
    NODE_CURRENT=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_CURRENT" -lt "$NODE_VERSION" ]; then
        log_warning "Node.js version $NODE_CURRENT is older than recommended $NODE_VERSION"
    fi
}

install_docker() {
    log_info "Installing Docker..."
    
    if ! check_command docker; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            log_info "Please install Docker Desktop for macOS from https://docker.com/products/docker-desktop"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Install Docker on Linux
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            
            # Install Docker Compose
            sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
        fi
    fi
}

setup_python_environment() {
    log_info "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3.11 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install project dependencies
    pip install -e ".[dev,gpu]"
    
    log_success "Python environment setup complete"
}

setup_frontend() {
    log_info "Setting up frontend..."
    
    cd ui
    
    # Install dependencies
    npm install
    
    # Build frontend
    npm run build
    
    cd ..
    
    log_success "Frontend setup complete"
}

setup_database() {
    log_info "Setting up database..."
    
    # Start PostgreSQL service
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start postgresql
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    fi
    
    # Create database and user
    sudo -u postgres psql -c "CREATE DATABASE openfire;" || true
    sudo -u postgres psql -c "CREATE USER openfire WITH PASSWORD 'openfire';" || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE openfire TO openfire;" || true
    
    log_success "Database setup complete"
}

setup_redis() {
    log_info "Setting up Redis..."
    
    # Start Redis service
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start redis
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start redis
        sudo systemctl enable redis
    fi
    
    log_success "Redis setup complete"
}

setup_environment_file() {
    log_info "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        cp env.example .env
        log_info "Created .env file from template"
        log_warning "Please edit .env file with your API keys and configuration"
    else
        log_info ".env file already exists"
    fi
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p models data logs static
    mkdir -p data/storage data/uploads data/exports
    mkdir -p logs/api logs/celery logs/nginx
    
    log_success "Directories created"
}

download_models() {
    log_info "Downloading AI models..."
    
    # Create models directory
    mkdir -p models
    
    # Download YOLOv8 base model (this would be replaced with actual fire detection model)
    python -c "
from ultralytics import YOLO
import os

model_path = 'models/yolov8n.pt'
if not os.path.exists(model_path):
    print('Downloading YOLOv8 model...')
    model = YOLO('yolov8n.pt')
    model.save(model_path)
    print('Model downloaded successfully')
else:
    print('Model already exists')
"
    
    log_success "Models downloaded"
}

initialize_database() {
    log_info "Initializing database..."
    
    source venv/bin/activate
    
    # Run database migrations
    python -c "
import asyncio
from openfire.database.connection import init_database

async def main():
    await init_database()
    print('Database initialized successfully')

asyncio.run(main())
"
    
    log_success "Database initialized"
}

setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Create monitoring directories
    mkdir -p monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources
    
    # Create Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'openfire-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
EOF
    
    log_success "Monitoring configuration created"
}

run_tests() {
    log_info "Running tests..."
    
    source venv/bin/activate
    
    # Run Python tests
    pytest --cov=openfire --cov-report=term-missing
    
    # Run frontend tests
    cd ui
    npm test -- --coverage --watchAll=false
    cd ..
    
    log_success "Tests completed"
}

start_services() {
    log_info "Starting services with Docker Compose..."
    
    # Start all services
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    docker-compose ps
    
    log_success "Services started"
}

print_summary() {
    log_success "OpenWildfires setup completed successfully!"
    echo
    echo "🚀 Quick Start:"
    echo "  • Web UI: http://localhost:3000"
    echo "  • API Docs: http://localhost:8000/docs"
    echo "  • Grafana: http://localhost:3001 (admin/admin)"
    echo "  • MLflow: http://localhost:5000"
    echo
    echo "📚 Next Steps:"
    echo "  1. Edit .env file with your API keys"
    echo "  2. Configure drone connections"
    echo "  3. Set up alert channels (Twilio, SendGrid)"
    echo "  4. Upload custom detection models"
    echo
    echo "🛠️ CLI Commands:"
    echo "  • openfire system status"
    echo "  • openfire drone list"
    echo "  • openfire detect image path/to/image.jpg"
    echo
    echo "📖 Documentation: https://docs.openwildfires.ai"
    echo "🐛 Issues: https://github.com/nikjois/openwildfires/issues"
    echo "💬 Support: nikjois@llamasearch.ai"
}

# Main execution
main() {
    echo "🔥 OpenWildfires Setup Script"
    echo "Author: Nik Jois <nikjois@llamasearch.ai>"
    echo "License: MIT"
    echo

    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        log_error "Please do not run this script as root"
        exit 1
    fi

    # System requirements check
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" != "darwin"* ]] && [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [ "$AVAILABLE_SPACE" -lt 10485760 ]; then  # 10GB in KB
        log_warning "Low disk space. At least 10GB recommended."
    fi
    
    # Installation steps
    install_python_dependencies
    install_node_dependencies
    install_docker
    
    setup_environment_file
    create_directories
    setup_python_environment
    setup_frontend
    setup_database
    setup_redis
    setup_monitoring
    download_models
    initialize_database
    
    # Optional: Run tests
    read -p "Run tests? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    # Optional: Start services
    read -p "Start services with Docker Compose? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_services
    fi
    
    print_summary
}

# Run main function
main "$@" 