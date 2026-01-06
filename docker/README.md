# Docker Setup for Heart Risk Prediction

This directory contains Docker configuration files for containerizing the Heart Risk Prediction application.

## Files

- **`Dockerfile`**: Container image definition for the application
- **`docker-compose.yml`**: Multi-service orchestration configuration
- **`entrypoint_app.sh`**: Application startup script
- **`requirements.txt`**: Docker-specific Python dependencies

## Quick Start

### Prerequisites
- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)

### Running the Application

1. **Build and start the application**:
   ```bash
   docker-compose up --build
   ```

2. **Access the services**:
   - **Gradio App**: http://localhost:7860
   - **Jupyter Notebooks**: http://localhost:8888

### Development Mode

For development with live code reloading:

```bash
docker-compose up --build -d
```

This runs the containers in detached mode, allowing you to continue working in the terminal.

### Stopping the Services

```bash
docker-compose down
```

## Service Details

### Heart Risk App Service
- **Port**: 7860
- **Function**: Runs the Gradio web interface
- **Volumes**: Maps data and results directories for persistence

### Jupyter Service  
- **Port**: 8888
- **Function**: Provides Jupyter notebook environment
- **Access**: No token required (development setup)
- **Volumes**: Maps entire project for notebook access

## Container Features

- **Python 3.9** runtime environment
- **All project dependencies** pre-installed
- **Automatic data preprocessing** on first run
- **Persistent data storage** via volume mounts
- **Hot reload** support for development

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Check what's using the port
   lsof -i :7860
   # Stop existing containers
   docker-compose down
   ```

2. **Build failures**:
   ```bash
   # Clean build (no cache)
   docker-compose build --no-cache
   ```

3. **Permission issues**:
   ```bash
   # Fix file permissions
   chmod +x entrypoint_app.sh
   ```

### Logs

View container logs:
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs heart-risk-app
docker-compose logs jupyter
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

For production deployment, consider:

1. **Remove Jupyter service** (security)
2. **Add authentication** to Gradio app
3. **Use environment variables** for secrets
4. **Configure proper logging**
5. **Set up health checks**
6. **Use production WSGI server**

Example production modifications:
```yaml
# Remove or comment out jupyter service
# Add environment variables
environment:
  - GRADIO_SERVER_NAME=0.0.0.0
  - GRADIO_AUTH_USERNAME=${AUTH_USERNAME}
  - GRADIO_AUTH_PASSWORD=${AUTH_PASSWORD}
```