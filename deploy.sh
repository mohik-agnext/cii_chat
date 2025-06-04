#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}Chandigarh Policy Assistant Deployment${NC}"
echo -e "${GREEN}=======================================${NC}"

# Check if .env file exists, if not create from template
if [ ! -f .env ]; then
    echo -e "\n${YELLOW}No .env file found. Creating from template...${NC}"
    cp env.docker .env
    echo -e "${YELLOW}Please edit the .env file with your API keys before continuing.${NC}"
    echo -e "Would you like to edit the .env file now? (y/n): "
    read edit_env
    if [[ $edit_env == "y" || $edit_env == "Y" ]]; then
        ${EDITOR:-nano} .env
    else
        echo -e "${YELLOW}Please remember to edit .env before starting the container.${NC}"
    fi
fi

# Build and start the Docker container
echo -e "\n${GREEN}Building and starting the Docker container...${NC}"
docker-compose -f hybrid-search-docker-compose.yml up -d --build

# Check if the container is running
echo -e "\n${GREEN}Checking container status...${NC}"
if docker-compose -f hybrid-search-docker-compose.yml ps | grep -q "Up"; then
    echo -e "${GREEN}✅ Container is running successfully!${NC}"
    
    # Get the container IP
    ip=$(hostname -I | awk '{print $1}')
    echo -e "\n${GREEN}Access the application at:${NC}"
    echo -e "http://${ip}:3000"
    
    echo -e "\n${GREEN}View logs with:${NC}"
    echo -e "docker-compose -f hybrid-search-docker-compose.yml logs -f"
else
    echo -e "${RED}❌ Container failed to start. Displaying logs...${NC}"
    docker-compose -f hybrid-search-docker-compose.yml logs
    exit 1
fi

echo -e "\n${GREEN}Deployment complete!${NC}"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "Railway CLI not found. Installing..."
    npm i -g @railway/cli
fi

# Login to Railway (if not already logged in)
railway login

# Deploy the application
echo "Deploying application to Railway..."
railway up

echo "Deployment complete!"
echo "Visit your Railway dashboard to check the status: https://railway.app/dashboard" 