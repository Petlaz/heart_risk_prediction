# Docker Setup for Heart Risk Prediction

**Status**: Docker configuration ready for deployment with professional medical interface and risk classification.

This directory contains Docker configuration files for containerizing the Heart Risk Assessment Platform with professional-grade medical styling, intelligent environment detection, and working Low/Moderate/High risk stratification.

## **Deployment Architecture**

**Docker Environment**: Professional medical interface on port 7860
**Local Environment**: Development interface on port 7861  
**No Port Conflicts**: Both can run simultaneously with identical styling
**Working Risk Classification**: Proper Low/Moderate/High stratification
**Medical-Grade Interface**: Healthcare industry styling and compliance

## Files

- **`Dockerfile`**: Container image definition for the application
- **`docker-compose.yml`**: Multi-service orchestration configuration
- **`entrypoint_app.sh`**: Professional application startup script with emoji logging
- **`requirements_docker.txt`**: Docker-optimized Python dependencies with version constraints
- **`README.md`**: This documentation file

## Quick Start

### Prerequisites
- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)

### Running the Application

1. **Build the application** (first time or after changes):
   ```bash
   cd /path/to/heart_risk_prediction
   docker-compose -f docker/docker-compose.yml build heart-risk-app
   ```

2. **Start the application**:
   ```bash
   docker-compose -f docker/docker-compose.yml up heart-risk-app
   ```

3. **Access the application**:
   - **Docker URL**: http://localhost:7860 (auto-detected Docker environment)
   - **Local Dev URL**: http://localhost:7861 (if running `python app/app_gradio.py` separately)
   - **Public URL**: Created automatically when share=True is enabled
   - **Smart Detection**: App automatically detects environment and uses appropriate port
   - **Professional Startup**: View emoji-enhanced logs during startup

4. **Stop the application**:
   ```bash
   docker-compose -f docker/docker-compose.yml down
   ```

### Alternative Deployment Methods

**Option 1: Docker Compose (Recommended for Production)**
```bash
docker-compose -f docker/docker-compose.yml up heart-risk-app
# Access: http://localhost:7860
```

**Option 2: Direct Docker Run**
```bash
docker build -t heart-risk-app -f docker/Dockerfile .
docker run -d --name heart-risk-app -p 7860:7860 heart-risk-app  
# Access: http://localhost:7860
```

**Option 3: Local Development (No Docker)**
```bash
cd /path/to/heart_risk_prediction
python app/app_gradio.py
# Access: http://localhost:7861 (auto-detected local environment)
```

**Option 4: Simultaneous Development & Production**
```bash
# Terminal 1: Start Docker production
docker-compose -f docker/docker-compose.yml up heart-risk-app

# Terminal 2: Start local development 
python app/app_gradio.py

# Result: Both running without conflicts
# Docker: http://localhost:7860
# Local:  http://localhost:7861
```

### Development Mode

For development with live code reloading:

```bash
docker-compose -f docker/docker-compose.yml up heart-risk-app -d
```

This runs the container in detached mode, allowing you to continue working in the terminal.

### Deployment Verification

**Check if the application is running**:
```bash
# Check running containers  
docker ps

# Test Docker application response (port 7860)
curl -s http://localhost:7860 | head -10

# Test Local application response (port 7861, if running)
curl -s http://localhost:7861 | head -10

# View Docker application logs
docker logs heart-risk-updated
```

**Expected startup logs**:

**Docker Environment:**
```
Starting Heart Risk Prediction Application...
Checking system requirements...
Processed data found
No trained models found - using fallback model
Starting Professional Heart Disease Risk Prediction App...
Detected Docker environment - using port 7860
Local URL: http://0.0.0.0:7860
Public URL: Will be created automatically with share=True
Docker deployment ready
```

**Local Environment:**
```
Loaded Adaptive Ensemble model: Adaptive_Ensemble_complexity_optimized_20260108_233028.joblib
Loaded preprocessing scaler
Loaded 22 feature names
Detected local environment - using port 7861
* Running on local URL: http://0.0.0.0:7861
* Running on public URL: https://371dc0fbe9eeba4b2a.gradio.live
```

### Stopping the Services

```bash
# Stop and remove containers
docker-compose -f docker/docker-compose.yml down

# Stop specific containers
docker stop heart-risk-updated  # Current container name
docker stop docker-heart-risk-app-1  # Legacy container name (if exists)
```

## Service Details

### Heart Risk App Service
- **Container**: heart-risk-updated (current deployment)
- **Legacy**: docker-heart-risk-app-1 (previous version)
- **Docker Port**: 7860 (auto-detected Docker environment)
- **Local Port**: 7861 (auto-detected local environment)
- **Function**: Runs the Gradio web interface with XAI capabilities
- **Smart Detection**: Automatically uses appropriate port based on environment
- **Features**: 
  - Professional startup logging with emojis
  - Environment auto-detection (Docker vs Local)
  - Automatic data preprocessing validation
  - Model detection and loading
  - SHAP and LIME explanations
  - Medical-grade professional interface
  - Risk stratification (Low/Moderate/High)
- **URLs**: 
  - Docker: http://localhost:7860
  - Local: http://localhost:7861  
  - Public: Auto-created with Gradio sharing
- **Volumes**: Maps data and results directories for persistence

### Dependencies
- **Python**: 3.9-slim runtime
- **ML Stack**: scikit-learn, numpy, pandas, joblib
- **Web Interface**: Gradio 3.50.0-4.0.0 (version constrained)
- **XAI Libraries**: SHAP, LIME
- **Visualization**: matplotlib, seaborn, plotly

## Container Features

- **Python 3.9** slim runtime environment
- **Version-constrained dependencies** for stability
- **Professional startup logging** with emoji status indicators
- **Automatic system validation** (data, models, preprocessing)
- **Both local and public URL access** capabilities
- **Heart Disease Risk Prediction** with XAI explanations
- **Adaptive Ensemble Model** with SHAP/LIME integration
- **Medical-grade interface** for clinical decision support
- **Persistent data storage** via volume mounts
- **Production-ready deployment** with Docker best practices

## Troubleshooting

### Common Issues

1. **Port already in use (7860)**:
   ```bash
   # Check what's using the port
   lsof -i :7860
   # Stop existing containers
   docker-compose -f docker/docker-compose.yml down
   # Kill any local Python processes
   pkill -f "python app/app_gradio.py"
   ```

2. **Build failures**:
   ```bash
   # Clean build (no cache)
   docker-compose -f docker/docker-compose.yml build --no-cache heart-risk-app
   ```

3. **Permission issues**:
   ```bash
   # Fix entrypoint permissions
   chmod +x docker/entrypoint_app.sh
   ```

4. **Container exits immediately**:
   ```bash
   # Check container logs
   docker logs docker-heart-risk-app-1
   # Verify data and models exist
   ls data/processed/
   ls results/models/
   ```

### Logs

View container logs:
```bash
# Current Heart Risk App logs
docker logs docker-heart-risk-app-1

# Follow logs in real-time
docker logs -f docker-heart-risk-app-1

# Docker Compose logs
docker-compose -f docker/docker-compose.yml logs heart-risk-app
```

## Customization

### Environment Variables

You can customize the setup by modifying environment variables in `docker-compose.yml`:

- **PYTHONPATH**: Python module search path
- **Custom settings**: Add your own environment variables as needed

### Volume Mounts

The default setup mounts:
- `../data:/app/data` - Data directory
- `../results:/app/results` - Results directory  
- `../:/app` - Full project (for Jupyter)

Modify volume mounts in `docker-compose.yml` as needed for your setup.

## Production Deployment

**Current Status**: Production-ready deployment achieved!

The current configuration includes:
- Professional startup logging and validation
- Version-constrained dependencies for stability  
- Both local and public URL capabilities
- Medical-grade XAI interface
- Proper Docker best practices

For enhanced production deployment, consider:

1. **Security enhancements**:
   ```yaml
   environment:
     - GRADIO_AUTH_USERNAME=${AUTH_USERNAME}
     - GRADIO_AUTH_PASSWORD=${AUTH_PASSWORD}
   ```

2. **Health checks**:
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:7860"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```

3. **Resource limits**:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: '1.0'
   ```

4. **Environment-specific configuration**:
   - Use secrets management for sensitive data
   - Configure proper logging levels
   - Set up monitoring and alerts
   - Implement backup strategies for models/data