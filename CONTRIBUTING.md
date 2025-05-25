# Contributing to OpenWildfires

Thank you for your interest in contributing to OpenWildfires! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to nikjois@llamasearch.ai.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git
- Basic knowledge of AI/ML, drone systems, or web development

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/openwildfires.git
   cd openwildfires
   ```

2. **Run the setup script**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Set up your development environment**
   ```bash
   # Python environment
   source venv/bin/activate
   pip install -e ".[dev]"
   
   # Frontend environment
   cd ui
   npm install
   cd ..
   ```

4. **Create your feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in existing code
- **Feature development**: Add new functionality
- **Documentation**: Improve or add documentation
- **Testing**: Add or improve tests
- **Performance**: Optimize existing code
- **Security**: Address security vulnerabilities
- **UI/UX**: Improve user interface and experience

### Areas of Focus

- **AI/ML Models**: Fire and smoke detection algorithms
- **Drone Integration**: MAVLink protocol, flight control
- **Computer Vision**: Image processing and analysis
- **Backend API**: FastAPI endpoints and business logic
- **Frontend UI**: React components and user experience
- **DevOps**: CI/CD, monitoring, deployment
- **Documentation**: Technical writing and tutorials

## Pull Request Process

### Before Submitting

1. **Check existing issues**: Look for related issues or discussions
2. **Create an issue**: For significant changes, create an issue first
3. **Follow coding standards**: Ensure your code follows our guidelines
4. **Write tests**: Add tests for new functionality
5. **Update documentation**: Update relevant documentation

### Submission Process

1. **Create a descriptive title**
   ```
   feat: add real-time fire spread prediction
   fix: resolve drone connection timeout issue
   docs: update API documentation for alerts
   ```

2. **Write a clear description**
   - What changes were made
   - Why the changes were necessary
   - How to test the changes
   - Any breaking changes

3. **Link related issues**
   ```
   Closes #123
   Fixes #456
   Related to #789
   ```

4. **Request review**
   - Tag relevant maintainers
   - Respond to feedback promptly
   - Make requested changes

### Review Criteria

Pull requests will be reviewed for:

- **Functionality**: Does it work as intended?
- **Code quality**: Is it well-written and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is it properly documented?
- **Performance**: Does it impact system performance?
- **Security**: Are there any security implications?

## Coding Standards

### Python

We follow PEP 8 with some modifications:

```python
# Use Black for formatting
black --line-length 88 .

# Use Ruff for linting
ruff check .

# Use MyPy for type checking
mypy openfire --ignore-missing-imports
```

**Key guidelines:**
- Use type hints for all functions
- Write docstrings for all public functions
- Use async/await for I/O operations
- Follow naming conventions (snake_case for functions/variables)
- Keep functions focused and small
- Use dataclasses or Pydantic models for data structures

**Example:**
```python
from typing import Optional, List
import asyncio
import structlog

logger = structlog.get_logger(__name__)

async def detect_fire(
    image: np.ndarray,
    confidence_threshold: float = 0.5
) -> Optional[DetectionResult]:
    """
    Detect fire in an image using AI models.
    
    Args:
        image: Input image as numpy array
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        Detection result or None if no fire detected
        
    Raises:
        ValueError: If image is invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided")
    
    try:
        # Detection logic here
        result = await model.detect(image)
        logger.info("Fire detection completed", confidence=result.confidence)
        return result
    except Exception as e:
        logger.error("Fire detection failed", error=str(e))
        raise
```

### TypeScript/React

We use TypeScript with strict mode enabled:

```typescript
// Use Prettier for formatting
prettier --write src/**/*.{ts,tsx}

// Use ESLint for linting
eslint src --ext .ts,.tsx
```

**Key guidelines:**
- Use TypeScript strict mode
- Define interfaces for all data structures
- Use functional components with hooks
- Follow React best practices
- Use meaningful component and variable names
- Write JSDoc comments for complex functions

**Example:**
```typescript
interface DroneStatus {
  id: string;
  isConnected: boolean;
  batteryLevel: number;
  location: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
}

interface DroneCardProps {
  drone: DroneStatus;
  onConnect: (droneId: string) => Promise<void>;
}

/**
 * Component for displaying drone status information
 */
const DroneCard: React.FC<DroneCardProps> = ({ drone, onConnect }) => {
  const [isConnecting, setIsConnecting] = useState(false);

  const handleConnect = async () => {
    setIsConnecting(true);
    try {
      await onConnect(drone.id);
    } catch (error) {
      console.error('Failed to connect to drone:', error);
    } finally {
      setIsConnecting(false);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6">{drone.id}</Typography>
        <Typography color={drone.isConnected ? 'success' : 'error'}>
          {drone.isConnected ? 'Connected' : 'Disconnected'}
        </Typography>
        <Button 
          onClick={handleConnect} 
          disabled={isConnecting || drone.isConnected}
        >
          {isConnecting ? 'Connecting...' : 'Connect'}
        </Button>
      </CardContent>
    </Card>
  );
};
```

### Git Commit Messages

We use conventional commits:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(detection): add ensemble model for fire detection
fix(drone): resolve connection timeout in MAVLink
docs(api): update OpenAPI specification
test(alerts): add unit tests for notification system
```

## Testing

### Test Requirements

All contributions must include appropriate tests:

- **Unit tests**: Test individual functions/components
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: For performance-critical code

### Running Tests

```bash
# Python tests
pytest --cov=openfire --cov-report=html

# Frontend tests
cd ui && npm test -- --coverage

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/ -m performance
```

### Test Guidelines

- Write tests before implementing features (TDD)
- Aim for >90% code coverage
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies
- Use fixtures for common test data

**Example test:**
```python
import pytest
from unittest.mock import Mock, patch
from openfire.detection import FireDetector

class TestFireDetector:
    @pytest.fixture
    def detector(self):
        return FireDetector(confidence_threshold=0.5)
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    async def test_detect_fire_success(self, detector, sample_image):
        """Test successful fire detection."""
        with patch.object(detector, 'model') as mock_model:
            mock_model.detect.return_value = Mock(confidence=0.85)
            
            result = await detector.detect(sample_image)
            
            assert result is not None
            assert result.confidence == 0.85
            mock_model.detect.assert_called_once()
    
    async def test_detect_fire_no_detection(self, detector, sample_image):
        """Test when no fire is detected."""
        with patch.object(detector, 'model') as mock_model:
            mock_model.detect.return_value = None
            
            result = await detector.detect(sample_image)
            
            assert result is None
    
    async def test_detect_fire_invalid_image(self, detector):
        """Test detection with invalid image."""
        with pytest.raises(ValueError, match="Invalid image"):
            await detector.detect(None)
```

## Documentation

### Documentation Types

- **API Documentation**: OpenAPI/Swagger specs
- **Code Documentation**: Docstrings and comments
- **User Documentation**: README, tutorials, guides
- **Developer Documentation**: Architecture, setup guides

### Writing Guidelines

- Use clear, concise language
- Include code examples
- Keep documentation up-to-date
- Use proper markdown formatting
- Include diagrams where helpful

### API Documentation

All API endpoints must be documented:

```python
@app.post("/detect/image", response_model=DetectionResponse)
async def detect_from_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user)
) -> DetectionResponse:
    """
    Detect fire and smoke in an uploaded image.
    
    This endpoint accepts an image file and returns detection results
    including bounding boxes, confidence scores, and AI analysis.
    
    Args:
        file: Image file (JPEG, PNG, WebP)
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        current_user: Authenticated user
        
    Returns:
        Detection results with fire/smoke locations and analysis
        
    Raises:
        HTTPException: 400 if image is invalid
        HTTPException: 401 if user is not authenticated
        HTTPException: 500 if detection fails
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/detect/image" \
          -H "Authorization: Bearer your-token" \
          -F "file=@wildfire.jpg" \
          -F "confidence_threshold=0.7"
        ```
    """
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear title**: Describe the issue briefly
2. **Environment**: OS, Python version, browser, etc.
3. **Steps to reproduce**: Detailed steps to recreate the issue
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Screenshots/logs**: If applicable
7. **Additional context**: Any other relevant information

**Template:**
```markdown
## Bug Description
Brief description of the bug

## Environment
- OS: macOS 13.0
- Python: 3.11.5
- Browser: Chrome 118.0
- OpenWildfires version: 1.0.0

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
A clear description of what actually happened.

## Screenshots
If applicable, add screenshots to help explain your problem.

## Additional Context
Add any other context about the problem here.
```

### Feature Requests

For feature requests, please include:

1. **Problem statement**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other solutions you've considered
4. **Use cases**: Who would benefit from this feature?
5. **Implementation notes**: Technical considerations

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: nikjois@llamasearch.ai for direct contact

### Getting Help

- Check existing documentation first
- Search existing issues and discussions
- Provide detailed information when asking questions
- Be respectful and patient

### Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes for significant contributions
- Annual contributor acknowledgments

## Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Individual feature branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation

### Release Process

1. **Feature freeze**: No new features in release branch
2. **Testing**: Comprehensive testing of release candidate
3. **Documentation**: Update documentation and changelog
4. **Tagging**: Create release tag with semantic versioning
5. **Deployment**: Deploy to production environment

### Semantic Versioning

We follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Security

### Reporting Security Issues

Please do not report security vulnerabilities through public GitHub issues. Instead, email nikjois@llamasearch.ai with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Guidelines

- Never commit secrets or API keys
- Use environment variables for configuration
- Follow secure coding practices
- Keep dependencies up to date
- Use HTTPS for all communications

## License

By contributing to OpenWildfires, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to OpenWildfires! Your efforts help make wildfire detection and response more effective, potentially saving lives and protecting communities.

For questions about contributing, please contact nikjois@llamasearch.ai or open a GitHub discussion. 