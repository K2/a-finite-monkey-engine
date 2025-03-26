#!/bin/bash

echo "Cleaning up project structure..."

# Create examples folder if it doesn't exist
mkdir -p ./examples/tests
mkdir -p ./examples/debug
mkdir -p ./examples/dev
mkdir -p ./examples/notebooks

# 1. Remove requirements.txt files that conflict with uv
echo "Removing requirements*.txt files..."
find . -name "requirements*.txt" -not -path "./examples/*" -type f -exec rm -v {} \;

# 2. Remove setup.py if it exists (not needed with uv)
if [ -f "setup.py" ]; then
    echo "Moving setup.py to examples/dev..."
    mv setup.py examples/dev/
fi

# 3. Move test files to examples/tests
echo "Moving test files to examples/tests..."
find . -name "*test*.py" -not -path "./examples/*" -not -path "./finite_monkey/test/*" -type f -exec mv -v {} ./examples/tests/ \;

# 4. Move development/debug utilities to examples
echo "Moving debug utilities to examples/debug..."
find . -name "*debug*.py" -not -path "./examples/*" -type f -exec mv -v {} ./examples/debug/ \;

# 5. Move notebooks to examples
echo "Moving notebooks to examples/notebooks..."
find . -name "*.ipynb" -not -path "./examples/*" -type f -exec mv -v {} ./examples/notebooks/ \;

# 6. Move starlight prototype files that aren't part of the core
echo "Moving starlight prototype files that aren't part of the core..."
mkdir -p ./examples/starlight-prototype/components
mkdir -p ./examples/starlight-prototype/pages
find ./starlight/src/pages -name "console.astro" -o -name "testing.astro" -exec mv -v {} ./examples/starlight-prototype/pages/ \;

# 7. Create pyproject.toml if it doesn't exist (for uv compatibility)
if [ ! -f "pyproject.toml" ]; then
    echo "Creating basic pyproject.toml for uv..."
    cat > pyproject.toml << EOF
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "finite-monkey-engine"
version = "0.1.0"
description = "A smart contract analysis framework"
readme = "README.md"
requires-python = ">=3.8"
license = { file = "LICENSE" }
authors = [
    { name = "Finite Monkey Team", email = "finitemodels@example.com" }
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["finite_monkey"]

EOF
    echo "Created pyproject.toml for uv compatibility"
fi

echo "Cleanup completed!"
