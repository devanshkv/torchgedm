# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-trixie

# Install system dependencies needed for building C++ extensions and f2c compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    f2c \
    libf2c2-dev \
    libopenblas-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Create a virtual environment
RUN uv venv

# Install setuptools explicitly (needed for pkg_resources)
RUN uv pip install setuptools

# Install dependencies from requirements.txt
RUN uv pip install -r requirements.txt

# Install test dependencies
RUN uv pip install -r requirements_test.txt

# Install the package in editable/development mode
RUN uv pip install -e .

# Activate the virtual environment by default
ENV PATH="/app/.venv/bin:$PATH"

# Make sure we're using the venv Python
ENV VIRTUAL_ENV="/app/.venv"


