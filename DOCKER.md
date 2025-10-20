# Docker Deployment Guide

This guide explains how to build and run the GIG Kuwait Claim Processing application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)
- Google Gemini API key

## Quick Start

### Using Docker Compose (Recommended)

1. **Create environment file**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your `GEMINI_API_KEY`

2. **Build and run**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   Open your browser and navigate to: `http://localhost:7860`

4. **Stop the application**
   ```bash
   docker-compose down
   ```

### Using Docker CLI

1. **Build the image**
   ```bash
   docker build -t gig-kwt-claim-processing .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name claim-processing-app \
     -p 7860:7860 \
     -e GEMINI_API_KEY=your_api_key_here \
     gig-kwt-claim-processing
   ```

3. **Access the application**
   Open your browser and navigate to: `http://localhost:7860`

4. **Stop the container**
   ```bash
   docker stop claim-processing-app
   docker rm claim-processing-app
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) | - |
| `PORT` | Application port | 7860 |

### Port Mapping

By default, the application runs on port 7860. You can change this by:

**Docker Compose:**
```yaml
ports:
  - "8080:7860"  # Maps host port 8080 to container port 7860
```

**Docker CLI:**
```bash
docker run -p 8080:7860 gig-kwt-claim-processing
```

## Monitoring

### View Logs

**Docker Compose:**
```bash
docker-compose logs -f
```

**Docker CLI:**
```bash
docker logs -f claim-processing-app
```

### Check Container Status

**Docker Compose:**
```bash
docker-compose ps
```

**Docker CLI:**
```bash
docker ps | grep claim-processing-app
```

### Health Check

The container includes a health check that runs every 30 seconds:
```bash
docker inspect --format='{{json .State.Health}}' claim-processing-app
```

## Production Deployment

### Resource Limits

Add resource limits to prevent container from consuming too many resources:

**docker-compose.yml:**
```yaml
services:
  claim-processing-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### HTTPS Setup

For production, use a reverse proxy like Nginx or Traefik to handle HTTPS:

**Example with Nginx:**
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Environment-Specific Deployment

Create different environment files:
- `.env.development`
- `.env.staging`
- `.env.production`

Then specify which to use:
```bash
docker-compose --env-file .env.production up -d
```

## Troubleshooting

### Container won't start

1. Check logs:
   ```bash
   docker-compose logs
   ```

2. Verify environment variables:
   ```bash
   docker-compose config
   ```

3. Ensure port 7860 is not already in use:
   ```bash
   netstat -tuln | grep 7860
   ```

### Application errors

1. Check if GEMINI_API_KEY is set correctly
2. Verify network connectivity
3. Check available disk space for temporary files

### Permission issues

If you encounter permission issues with mounted volumes:
```bash
docker-compose down
docker-compose up --build
```

## Updating the Application

1. Pull latest changes
2. Rebuild the image:
   ```bash
   docker-compose build --no-cache
   ```
3. Restart the container:
   ```bash
   docker-compose up -d
   ```

## Cleanup

### Remove containers and images

```bash
# Stop and remove containers
docker-compose down

# Remove the image
docker rmi gig-kwt-claim-processing

# Remove unused images and volumes
docker system prune -a
```

## Security Best Practices

1. Never commit `.env` file to version control
2. Use Docker secrets for sensitive data in production
3. Regularly update base images
4. Run containers with non-root user (consider adding USER directive in Dockerfile)
5. Scan images for vulnerabilities:
   ```bash
   docker scan gig-kwt-claim-processing
   ```

## Support

For issues or questions, please refer to the main README.md or contact the development team.
