#!/bin/bash

# Script to push your audio-classifier project to GitHub for Colab training

echo "🚀 Preparing to push audio-classifier to GitHub..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Initializing..."
    git init
    echo "📝 Please add your GitHub remote:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/audio-classifier.git"
    exit 1
fi

# Add all files
echo "📁 Adding all files..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Add cloud training support and Colab notebook

- Add cloud-optimized training script with auto-detection
- Add comprehensive Colab setup notebook
- Add cloud deployment guide and requirements
- Add automated setup scripts for different platforms
- Optimize configurations for various cloud environments"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main

echo "✅ Successfully pushed to GitHub!"
echo ""
echo "🎯 Next steps for Colab:"
echo "1. Go to https://colab.research.google.com/"
echo "2. Upload the colab_setup.ipynb file"
echo "3. Update the GitHub URL in cell 2 with your actual repo URL"
echo "4. Enable GPU: Runtime → Change runtime type → GPU"
echo "5. Run all cells!"
