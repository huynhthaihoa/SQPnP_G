#!/bin/bash

# SQPnP Repository Update Script
# This script helps update your GitHub repository with Python bindings and new documentation

set -e  # Exit on any error

echo "🚀 SQPnP Repository Update Script"
echo "=================================="
echo ""

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo "❌ Error: Not in a git repository"
        echo "Please run this script from the root of your SQPnP repository"
        exit 1
    fi
    echo "✅ Git repository detected"
}

# Function to check git status
check_git_status() {
    echo ""
    echo "📊 Current Git Status:"
    git status --short
    
    echo ""
    echo "📝 Recent commits:"
    git log --oneline -5
}

# Function to clean compiled files
clean_compiled_files() {
    echo ""
    echo "🧹 Cleaning compiled files..."
    
    # Remove compiled Python files
    find . -name "*.so" -delete
    find . -name "*.pyd" -delete
    find . -name "*.o" -delete
    find . -name "*.a" -delete
    
    # Remove Python cache
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete
    
    # Remove build directories
    rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
    
    echo "✅ Compiled files cleaned"
}

# Function to add new files
add_new_files() {
    echo ""
    echo "📁 Adding new files to git..."
    
    # Add all new files
    git add .
    
    echo "✅ Files added to staging area"
}

# Function to show what will be committed
show_changes() {
    echo ""
    echo "📋 Changes to be committed:"
    git diff --cached --name-status
    
    echo ""
    echo "📄 Summary of changes:"
    echo "- Updated main README.md with comprehensive documentation"
    echo "- Added Python bindings with pybind11"
    echo "- Created comprehensive .gitignore file"
    echo "- Updated Python bindings README with detailed instructions"
    echo "- Added installation script for Python bindings"
    echo "- Updated requirements.txt with all dependencies"
    echo "- Enhanced example usage with comprehensive tests"
    echo "- Added real-time head pose estimation demo"
    echo "- Added demo screenshots in assests/ folder"
}

# Function to commit changes
commit_changes() {
    echo ""
    echo "💾 Committing changes..."
    
    # Create commit message
    commit_msg="feat: Add Python bindings and comprehensive documentation

- Add Python bindings using pybind11 for easy integration
- Create comprehensive README.md with C++ and Python usage
- Add detailed Python bindings documentation and examples
- Include real-time head pose estimation demo
- Add installation script for automated setup
- Update project structure and add .gitignore
- Include demo screenshots and assets
- Add comprehensive testing and benchmarking examples

Python bindings features:
- Support for pinhole, fisheye, and distortion camera models
- Integrated RANSAC for robust estimation
- High-performance C++ bindings with minimal overhead
- Real-time head pose estimation with MediaPipe integration
- Comprehensive error handling and validation

Breaking changes: None
Migration guide: See README.md for installation instructions"
    
    git commit -m "$commit_msg"
    
    echo "✅ Changes committed successfully"
}

# Function to push to remote
push_changes() {
    echo ""
    echo "🚀 Pushing to remote repository..."
    
    # Get current branch
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"
    
    # Ask user if they want to push
    read -p "Do you want to push to remote repository? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin "$current_branch"
        echo "✅ Changes pushed to remote repository"
    else
        echo "⏭️  Skipping push to remote"
    fi
}

# Function to create release notes
create_release_notes() {
    echo ""
    echo "📝 Creating release notes..."
    
    cat > RELEASE_NOTES.md << 'EOF'
# SQPnP v2.0.0 - Python Bindings Release

## 🎉 What's New

This major release adds comprehensive Python bindings to the SQPnP library, making it accessible to a wider audience of researchers and developers.

### ✨ New Features

#### Python Bindings
- **Easy-to-use Python API**: Simple interface with NumPy arrays
- **Multiple camera models**: Support for pinhole, fisheye, and distortion models
- **Integrated RANSAC**: Robust estimation with outlier rejection
- **High performance**: Direct C++ bindings with minimal overhead
- **Real-time demo**: Live head pose estimation with webcam

#### Documentation & Examples
- **Comprehensive README**: Updated with both C++ and Python usage
- **Detailed API reference**: Complete documentation for all features
- **Installation scripts**: Automated setup for different platforms
- **Real-time demo**: Head pose estimation with MediaPipe integration
- **Performance benchmarks**: Comprehensive testing and optimization

#### Developer Experience
- **Automated installation**: Cross-platform install script
- **Comprehensive testing**: Unit tests and integration tests
- **Error handling**: Robust validation and error reporting
- **Performance optimization**: Optimized for real-time applications

### 🔧 Installation

#### C++ (Existing)
```bash
mkdir build && cd build
cmake ..
make -j4
```

#### Python (New)
```bash
cd python_bindings
pip install -e .
```

### 📖 Quick Start

#### Python Usage
```python
import numpy as np
import sqpnp_python

# Create solver and solve PnP
solver = sqpnp_python.SQPnPSolver()
result = solver.solve_pinhole(points_3d, points_2d, camera_params)

if result.success:
    print(f"Rotation: {result.rotation}")
    print(f"Translation: {result.translation}")
```

#### Real-time Demo
```bash
python adapter.py
```

### 🎮 Demo Features

- Real-time face detection using MediaPipe
- 3D head pose estimation using SQPnP
- Visual axes overlay on the face
- Performance metrics (FPS, CPU usage)
- Support for different landmark subsets

### 📊 Performance

- **C++**: Optimized for embedded devices and real-time applications
- **Python**: Minimal overhead with direct C++ bindings
- **RANSAC**: Integrated outlier rejection for robust estimation
- **Multi-threading**: Support for parallel processing

### 🔄 Migration Guide

No breaking changes for existing C++ users. Python users can start with the new bindings immediately.

### 📚 Documentation

- [Main README](README.md) - Comprehensive project overview
- [C++ API Documentation](CLEAN_README.md) - Detailed C++ usage
- [Python API Documentation](python_bindings/README.md) - Python bindings guide
- [Real-time Demo](adapter.py) - Live head pose estimation

### 🙏 Acknowledgments

- Original SQPnP implementation by G. Terzakis and M. Lourakis
- RANSAC implementation from RansacLib
- Python bindings using pybind11
- Real-time demo using MediaPipe Face Mesh

### 📄 License

Same as the original SQPnP project.

---

**Note**: This release maintains full backward compatibility with existing C++ code while adding powerful new Python capabilities.
EOF

    echo "✅ Release notes created in RELEASE_NOTES.md"
}

# Function to show next steps
show_next_steps() {
    echo ""
    echo "🎯 Next Steps:"
    echo "=============="
    echo ""
    echo "1. 📝 Review the changes:"
    echo "   git diff HEAD~1"
    echo ""
    echo "2. 🧪 Test the installation:"
    echo "   cd python_bindings && python test_sqpnp.py"
    echo ""
    echo "3. 🎮 Try the real-time demo:"
    echo "   python adapter.py"
    echo ""
    echo "4. 📚 Update your GitHub repository:"
    echo "   - Create a new release with RELEASE_NOTES.md"
    echo "   - Update repository description and topics"
    echo "   - Add badges for build status and Python version"
    echo ""
    echo "5. 📖 Update external documentation:"
    echo "   - Update any external references to your repository"
    echo "   - Share with the computer vision community"
    echo ""
    echo "6. 🔧 Consider additional improvements:"
    echo "   - Add CI/CD pipeline for automated testing"
    echo "   - Create conda package for easier distribution"
    echo "   - Add more camera models and features"
    echo ""
    echo "✅ Repository update completed successfully!"
}

# Main execution
main() {
    echo "Starting repository update process..."
    echo ""
    
    # Check if we're in a git repository
    check_git_repo
    
    # Show current status
    check_git_status
    
    # Clean compiled files
    clean_compiled_files
    
    # Add new files
    add_new_files
    
    # Show changes
    show_changes
    
    # Ask for confirmation
    echo ""
    read -p "Do you want to commit these changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Commit changes
        commit_changes
        
        # Create release notes
        create_release_notes
        
        # Push to remote
        push_changes
        
        # Show next steps
        show_next_steps
    else
        echo "⏭️  Update cancelled"
        echo "You can manually commit the changes later with:"
        echo "  git add ."
        echo "  git commit -m 'Your commit message'"
        echo "  git push origin main"
    fi
}

# Run main function
main "$@" 