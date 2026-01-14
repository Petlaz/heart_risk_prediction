# Docker Setup for Heart Risk Prediction

✅ **Status**: Successfully deployed and running!

This directory contains Docker configuration files for containerizing the Heart Risk Prediction application with professional-grade startup logging and both local and public URL capabilities.

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
   - **Local URL**: http://localhost:7860
   - **Public URL**: Generated automatically when share=True is enabled
   - **Professional Startup**: View emoji-enhanced logs during startup

4. **Stop the application**:
   ```bash
   docker-compose -f docker/docker-compose.yml down
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

# Test application response
curl -s http://localhost:7860 | head -10

# View application logs
docker logs docker-heart-risk-app-1
```

**Expected startup logs**:
```
🫀 Starting Heart Risk Prediction Application...
📋 Checking system requirements...
✅ Processed data found
✅ Trained models found  
🚀 Starting Professional Heart Disease Risk Prediction App...
📱 Local URL: http://0.0.0.0:7860
🌐 Public URL: Will be generated automatically with share=True
🐳 Docker deployment ready
```

### Stopping the Services

```bash
# Stop and remove containers
docker-compose -f docker/docker-compose.yml down

# Stop all related containers (if needed)
docker stop docker-heart-risk-app-1
```

## Service Details

### Heart Risk App Service
- **Container**: docker-heart-risk-app-1
- **Port**: 7860 (0.0.0.0:7860->7860/tcp)
- **Function**: Runs the Gradio web interface with XAI capabilities
- **Features**: 
  - Professional startup logging with emojis
  - Automatic data preprocessing validation
  - Model detection and loading
  - SHAP and LIME explanations
  - Medical-grade interface
- **URLs**: 
  - Local: http://localhost:7860
  - Public: Auto-generated with Gradio sharing
- **Volumes**: Maps data and results directories for persistence

### Dependencies
- **Python**: 3.9-slim runtime
- **ML Stack**: scikit-learn, numpy, pandas, joblib
- **Web Interface**: Gradio 3.50.0-4.0.0 (version constrained)
- **XAI Libraries**: SHAP, LIME
- **Visualization**: matplotlib, seaborn, plotly

## Container Features

- ✅ **Python 3.9** slim runtime environment
- ✅ **Version-constrained dependencies** for stability
- ✅ **Professional startup logging** with emoji status indicators
- ✅ **Automatic system validation** (data, models, preprocessing)
- ✅ **Both local and public URL access** capabilities
- ✅ **Heart Disease Risk Prediction** with XAI explanations
- ✅ **Adaptive Ensemble Model** with SHAP/LIME integration
- ✅ **Medical-grade interface** for clinical decision support
- ✅ **Persistent data storage** via volume mounts
- ✅ **Production-ready deployment** with Docker best practices

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

✅ **Current Status**: Production-ready deployment achieved!

The current configuration includes:
- ✅ Professional startup logging and validation
- ✅ Version-constrained dependencies for stability  
- ✅ Both local and public URL capabilities
- ✅ Medical-grade XAI interface
- ✅ Proper Docker best practices

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