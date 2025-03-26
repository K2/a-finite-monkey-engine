#!/bin/bash

# Create required directories
mkdir -p /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/components
mkdir -p /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/pages
mkdir -p /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/utils
mkdir -p /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/api
mkdir -p /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/layouts

# Move prototype files that have been superseded

# Move prototype pages that aren't part of the main flow
mv /home/files/git/a-finite-monkey-engine/starlight/src/pages/flow-analyzer.astro /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/pages/
mv /home/files/git/a-finite-monkey-engine/starlight/src/pages/contracts.astro /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/pages/
mv /home/files/git/a-finite-monkey-engine/starlight/src/pages/testing.astro /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/pages/
mv /home/files/git/a-finite-monkey-engine/starlight/src/pages/security.astro /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/pages/
mv /home/files/git/a-finite-monkey-engine/starlight/src/pages/console.astro /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/pages/

# Move components that have been replaced with secure versions
mv /home/files/git/a-finite-monkey-engine/starlight/src/components/CollaborationPanel.jsx /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/components/
mv /home/files/git/a-finite-monkey-engine/starlight/src/components/ChatInterface.jsx /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/components/
mv /home/files/git/a-finite-monkey-engine/starlight/src/components/PyConsole.jsx /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/components/

# Move utilities that have been superseded by enhanced versions
mv /home/files/git/a-finite-monkey-engine/starlight/src/utils/translation.ts /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/utils/
mv /home/files/git/a-finite-monkey-engine/starlight/src/utils/collaboration.ts /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/utils/

# Move API endpoints that have been replaced
mv /home/files/git/a-finite-monkey-engine/starlight/src/pages/api/translate.js /home/files/git/a-finite-monkey-engine/examples/starlight-prototype/api/
