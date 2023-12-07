#!/bin/bash

# Navigate to your project repository
cd "/home/sayem/Desktop/SODIndoorLoc"

# Define the remote repository URL
REPO_URL="git@github.com:skhan61/SODIndoorLoc.git"

# Pull the latest changes from the remote repository
git pull origin master

# Add all changes in the root directory, excluding those in .gitignore
git add .

# Explicitly add changes from each subdirectory
git add notebooks/* src/dataset/* src/trainers/* src/models/* src/experiments/* 

# Commit the changes with the current date and time
if [ -n "$(git diff)" ] || [ -n "$(git diff --cached)" ]; then
    git commit -m "Automatic commit at $(date)"

    # Push changes to the remote repository
    git push origin master
else
    echo "No changes detected."
fi
