#!/bin/bash

# SQPnP Python Bindings Installation Script
# This script installs the Python bindings for SQPnP

set -e  # Exit on any error

echo "=== SQPnP Python Bindings Installation ==="
echo ""

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system dependencies
install_system_deps() {
    local os=$(detect_os)
    
    echo "Installing system dependencies for $os..."
    
    case $os in
        "linux")
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y build-essential cmake libeigen3-dev
            elif command_exists yum; then
                sudo yum install -y gcc-c++ cmake eigen3-devel
            elif command_exists dnf; then
                sudo dnf install -y gcc-c++ cmake eigen3-devel
            else
                echo "Warning: Could not install system dependencies automatically"
                echo "Please install: build-essential, cmake, libeigen3-dev"
            fi
            ;;
        "macos")
            if command_exists brew; then
                brew install eigen cmake
            else
                echo "Warning: Homebrew not found. Please install Homebrew and run:"
                echo "brew install eigen cmake"
            fi
            ;;
        "windows")
            echo "For Windows, please install:"
            echo "1. Visual Studio Build Tools or Visual Studio Community"
            echo "2. Eigen3 (via vcpkg: vcpkg install eigen3)"
            echo "3. CMake"
            ;;
        *)
            echo "Unknown OS. Please install dependencies manually."
            ;;
    esac
}

# Function to check Python version
check_python() {
    if ! command_exists python3; then
        echo "Error: Python 3 is required but not found"
        exit 1
    fi
    
    local version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Found Python $version"
    
    # Check if version is >= 3.7
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 7 ]); then
        echo "Error: Python 3.7 or higher is required"
        exit 1
    fi
}

# Function to install Python dependencies
install_python_deps() {
    echo "Installing Python dependencies..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install core dependencies
    python3 -m pip install pybind11 numpy
    
    # Install optional dependencies for demo
    python3 -m pip install opencv-python mediapipe
    
    echo "Python dependencies installed successfully"
}

# Function to build and install the package
build_package() {
    echo "Building and installing SQPnP Python bindings..."
    
    # Clean any previous builds
    rm -rf build/ *.so *.pyd dist/ *.egg-info/ 2>/dev/null || true
    
    # Install in development mode
    python3 -m pip install -e .
    
    echo "Build completed successfully"
}

# Function to test installation
test_installation() {
    echo "Testing installation..."
    
    if python3 -c "import sqpnp_python; print('SQPnP Python bindings imported successfully')"; then
        echo "✅ Installation test passed"
    else
        echo "❌ Installation test failed"
        exit 1
    fi
    
    # Run the test script if it exists
    if [ -f "test_sqpnp.py" ]; then
        echo "Running comprehensive tests..."
        python3 test_sqpnp.py
    fi
}

# Main installation process
main() {
    echo "Starting installation process..."
    echo ""
    
    # Check Python
    check_python
    echo ""
    
    # Install system dependencies
    install_system_deps
    echo ""
    
    # Install Python dependencies
    install_python_deps
    echo ""
    
    # Build and install package
    build_package
    echo ""
    
    # Test installation
    test_installation
    echo ""
    
    echo "=== Installation completed successfully! ==="
    echo ""
    echo "You can now use SQPnP Python bindings:"
    echo "  import sqpnp_python"
    echo ""
    echo "To run the real-time demo:"
    echo "  cd .. && python adapter.py"
    echo ""
    echo "For more information, see the README.md file"
}

# Run main function
main "$@" 