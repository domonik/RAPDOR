#!/bin/bash

# Find the latest version directory
latest_version=$(find ./build/html -maxdepth 1 -type d -name 'v*' | sort -V | tail -n 1)

# Strip the prefix '/build/html'
latest_version=${latest_version#./build/html/}
# Replace the placeholder in the HTML file
sed "s#{{ version }}#$latest_version#g" superindex.html > build/html/index.html