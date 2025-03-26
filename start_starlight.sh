#!/bin/bash

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARLIGHT_DIR="${SCRIPT_DIR}/starlight"

# Default values
USE_BUN=false
NODE_VERSION="18" # Using a more stable version
DEV_MODE=true
CLEAN_INSTALL=false
DEBUG=true # Enable debug mode by default for troubleshooting
OPEN_BROWSER=true
VERBOSE=false

# Print help message
print_help() {
  echo -e "${BLUE}Starlight UI Starter Script${NC}"
  echo -e "Usage: ./start_starlight.sh [options]"
  echo ""
  echo "Options:"
  echo "  -b, --bun                Use Bun instead of Node.js (if available)"
  echo "  -n, --node <version>     Specify Node.js version to use with nvm (default: $NODE_VERSION)"
  echo "  -p, --production         Start in production mode (default: development mode)"
  echo "  -c, --clean              Perform a clean install (delete node_modules)"
  echo "  -d, --debug              Start with debugging enabled"
  echo "  -v, --verbose            Show verbose output for troubleshooting"
  echo "  --no-open                Don't open browser automatically"
  echo "  -h, --help               Show this help message"
  echo ""
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -b|--bun) USE_BUN=true ;;
    -n|--node) NODE_VERSION="$2"; shift ;;
    -p|--production) DEV_MODE=false ;;
    -c|--clean) CLEAN_INSTALL=true ;;
    -d|--debug) DEBUG=true ;;
    -v|--verbose) VERBOSE=true ;;
    --no-open) OPEN_BROWSER=false ;;
    -h|--help) print_help; exit 0 ;;
    *) echo -e "${RED}Unknown parameter: $1${NC}"; print_help; exit 1 ;;
  esac
  shift
done

# Verify directory structure
echo -e "${BLUE}Verifying Starlight directory structure...${NC}"
mkdir -p "${STARLIGHT_DIR}/src/pages"
mkdir -p "${STARLIGHT_DIR}/src/components"
mkdir -p "${STARLIGHT_DIR}/src/layouts"
mkdir -p "${STARLIGHT_DIR}/public"

# Check if we're in the right directory
if [ ! -d "$STARLIGHT_DIR" ]; then
  echo -e "${RED}Error: Starlight directory not found at $STARLIGHT_DIR${NC}"
  echo -e "${YELLOW}Creating Starlight directory structure...${NC}"
  mkdir -p "$STARLIGHT_DIR"
  mkdir -p "${STARLIGHT_DIR}/src/pages"
  mkdir -p "${STARLIGHT_DIR}/src/components"
  mkdir -p "${STARLIGHT_DIR}/src/layouts"
  mkdir -p "${STARLIGHT_DIR}/public"
  
  # Create minimal configuration files if needed
  if [ ! -f "${STARLIGHT_DIR}/package.json" ]; then
    echo -e "${YELLOW}Creating minimal package.json...${NC}"
    echo '{
  "name": "starlight-ui",
  "type": "module",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "start": "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  },
  "dependencies": {
    "@astrojs/react": "^3.0.2",
    "@astrojs/node": "^6.0.0",
    "astro": "^3.1.1",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}' > "${STARLIGHT_DIR}/package.json"
  fi
  
  if [ ! -f "${STARLIGHT_DIR}/astro.config.mjs" ]; then
    echo -e "${YELLOW}Creating minimal astro.config.mjs...${NC}"
    echo "import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import node from '@astrojs/node';

export default defineConfig({
  integrations: [react()],
  output: 'server',
  adapter: node({
    mode: 'standalone'
  }),
  server: {
    port: 3000,
    host: true
  }
});" > "${STARLIGHT_DIR}/astro.config.mjs"
  fi
  
  if [ ! -f "${STARLIGHT_DIR}/src/pages/index.astro" ]; then
    echo -e "${YELLOW}Creating minimal index page...${NC}"
    echo '---
---
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Starlight UI</title>
  <style>
    body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 2rem; }
    h1 { color: #3b82f6; }
  </style>
</head>
<body>
  <h1>Starlight UI is running!</h1>
  <p>🐒 Minimal test page</p>
</body>
</html>' > "${STARLIGHT_DIR}/src/pages/index.astro"
  fi
fi

# Navigate to Starlight directory
cd "$STARLIGHT_DIR" || { echo -e "${RED}Failed to navigate to $STARLIGHT_DIR${NC}"; exit 1; }
echo -e "${GREEN}Successfully navigated to $STARLIGHT_DIR${NC}"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check package.json exists
if [ ! -f "package.json" ]; then
  echo -e "${RED}Error: package.json not found${NC}"
  echo -e "${YELLOW}Creating a minimal package.json${NC}"
  echo '{
  "name": "starlight-ui",
  "type": "module",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  }
}' > package.json
fi

# Setup NVM if we're using Node
if [ "$USE_BUN" = false ]; then
  # Check for NVM
  if [ -f "$HOME/.nvm/nvm.sh" ]; then
    echo -e "${BLUE}Loading NVM...${NC}"
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
    
    # Install and use the specified Node.js version
    echo -e "${BLUE}Setting up Node.js v$NODE_VERSION...${NC}"
    nvm install "$NODE_VERSION" || { echo -e "${RED}Failed to install Node.js v$NODE_VERSION${NC}"; exit 1; }
    nvm use "$NODE_VERSION" || { echo -e "${RED}Failed to use Node.js v$NODE_VERSION${NC}"; exit 1; }
  else
    echo -e "${YELLOW}NVM not found, using system Node.js...${NC}"
    if ! command_exists node; then
      echo -e "${RED}Node.js is not installed. Please install Node.js or setup NVM.${NC}"
      exit 1
    fi
  fi
else
  # Check for Bun
  if (! command_exists bun); then
    echo -e "${RED}Bun is not installed. Please install Bun or use Node.js instead.${NC}"
    exit 1
  fi
  echo -e "${BLUE}Using Bun runtime...${NC}"
fi

# Show environment information
if [ "$VERBOSE" = true ]; then
  echo -e "${BLUE}Environment information:${NC}"
  echo -e "Node.js version: $(node --version)"
  echo -e "NPM version: $(npm --version)"
  if command_exists bun; then
    echo -e "Bun version: $(bun --version)"
  fi
  echo -e "Current directory: $(pwd)"
  echo -e "Files in directory:"
  ls -la
  echo -e "Package.json content:"
  cat package.json
  if [ -f "astro.config.mjs" ]; then
    echo -e "astro.config.mjs content:"
    cat astro.config.mjs
  else
    echo -e "${YELLOW}astro.config.mjs not found${NC}"
  fi
fi

# Clean install if requested
if [ "$CLEAN_INSTALL" = true ]; then
  echo -e "${YELLOW}Performing clean install...${NC}"
  if [ -d "node_modules" ]; then
    rm -rf node_modules
    echo -e "${GREEN}Removed node_modules directory${NC}"
  fi
  
  # Also remove lockfiles for clean start
  if [ -f "package-lock.json" ]; then
    rm package-lock.json
    echo -e "${GREEN}Removed package-lock.json${NC}"
  fi
  if [ -f "yarn.lock" ]; then
    rm yarn.lock
    echo -e "${GREEN}Removed yarn.lock${NC}"
  fi
  if [ -f "pnpm-lock.yaml" ]; then
    rm pnpm-lock.yaml
    echo -e "${GREEN}Removed pnpm-lock.yaml${NC}"
  fi
  if [ -f "bun.lockb" ]; then
    rm bun.lockb
    echo -e "${GREEN}Removed bun.lockb${NC}"
  fi
fi

# Install missing dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
MISSING_DEPS=false

# Check for required dependencies and install them if missing
if ! grep -q '"astro":' package.json; then
  echo -e "${YELLOW}Astro dependency missing, adding it...${NC}"
  MISSING_DEPS=true
fi

if ! grep -q '"@astrojs/react":' package.json; then
  echo -e "${YELLOW}@astrojs/react dependency missing, adding it...${NC}"
  MISSING_DEPS=true
fi

if ! grep -q '"@astrojs/node":' package.json; then
  echo -e "${YELLOW}@astrojs/node dependency missing, adding it...${NC}"
  MISSING_DEPS=true
fi

if [ "$MISSING_DEPS" = true ] || [ ! -d "node_modules" ]; then
  echo -e "${BLUE}Installing dependencies...${NC}"
  if [ "$USE_BUN" = true ]; then
    bun install || { echo -e "${RED}Failed to install dependencies with Bun${NC}"; exit 1; }
  else
    npm install || { echo -e "${RED}Failed to install dependencies with npm${NC}"; exit 1; }
    
    # Also install specific dependencies if they're missing
    if ! grep -q '"astro":' package.json; then
      npm install astro@3.1.1 || { echo -e "${RED}Failed to install astro${NC}"; exit 1; }
    fi
    if ! grep -q '"@astrojs/react":' package.json; then
      npm install @astrojs/react@3.0.2 || { echo -e "${RED}Failed to install @astrojs/react${NC}"; exit 1; }
    fi
    if ! grep -q '"@astrojs/node":' package.json; then
      npm install @astrojs/node@6.0.0 || { echo -e "${RED}Failed to install @astrojs/node${NC}"; exit 1; }
    fi
    if ! grep -q '"react":' package.json; then
      npm install react@18.2.0 react-dom@18.2.0 || { echo -e "${RED}Failed to install react${NC}"; exit 1; }
    fi
  fi
  echo -e "${GREEN}Dependencies installed successfully${NC}"
fi

# Setup environment variables for debugging
if [ "$DEBUG" = true ]; then
  export DEBUG="genaiscript:*,astro:*"
  export ASTRO_LOG_LEVEL="debug"
  echo -e "${BLUE}Debug mode enabled${NC}"
fi

# Start the application
echo -e "${GREEN}Starting Starlight UI...${NC}"
if [ "$VERBOSE" = true ]; then
  echo -e "${BLUE}Starting with command:${NC}"
  if [ "$DEV_MODE" = true ]; then
    if [ "$USE_BUN" = true ]; then
      BROWSER_FLAG=""
      if [ "$OPEN_BROWSER" = false ]; then
        BROWSER_FLAG="--no-open"
      fi
      echo "bun run dev $BROWSER_FLAG"
    else
      if [ "$OPEN_BROWSER" = false ]; then
        echo "BROWSER=none npm run dev"
      else
        echo "npm run dev"
      fi
    fi
  else
    if [ "$USE_BUN" = true ]; then
      echo "bun run build && bun run preview"
    else
      echo "npm run build && npm run preview"
    fi
  fi
fi

if [ "$DEV_MODE" = true ]; then
  # Development mode
  if [ "$USE_BUN" = true ]; then
    BROWSER_FLAG=""
    if [ "$OPEN_BROWSER" = false ];then
      BROWSER_FLAG="--no-open"
    fi
    bun run dev $BROWSER_FLAG
  else
    if [ "$OPEN_BROWSER" = false ]; then
      BROWSER=none npm run dev
    else
      npm run dev
    fi
  fi
else
  # Production mode
  if [ "$USE_BUN" = true ]; then
    bun run build && bun run preview
  else
    npm run build && npm run preview
  fi
fi
