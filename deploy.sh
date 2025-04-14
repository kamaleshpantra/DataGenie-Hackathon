#!/bin/bash
# Push to GitHub
git add .
git commit -m "Final submission for DataGenie Hackathon 2025"
git push origin main

# Deploy to Render (configure .render.yaml separately)
echo "Deploying to Render. Check Render dashboard."