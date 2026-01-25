#!/bin/bash
# Helper script to set HuggingFace token as environment variable

echo "=========================================="
echo "HuggingFace Token Setup"
echo "=========================================="
echo ""
echo "To create a token:"
echo "1. Go to: https://huggingface.co/settings/tokens"
echo "2. Click 'New token'"
echo "3. Name it (e.g., 'topic-modeling-project')"
echo "4. Select 'Read' access"
echo "5. Click 'Generate token'"
echo "6. Copy the token"
echo ""
echo "=========================================="
echo ""

# Prompt for token
read -sp "Paste your HuggingFace token here: " HF_TOKEN
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "Error: Token cannot be empty"
    exit 1
fi

# Set for current session
export HF_TOKEN="$HF_TOKEN"
echo "✓ Token set for current terminal session"

# Ask if user wants to make it permanent
read -p "Do you want to add this to ~/.zshrc for permanent use? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if already exists in .zshrc
    if grep -q "HF_TOKEN" ~/.zshrc 2>/dev/null; then
        echo "⚠ HF_TOKEN already exists in ~/.zshrc"
        read -p "Do you want to replace it? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Remove old entry
            sed -i.bak '/^export HF_TOKEN=/d' ~/.zshrc
            echo "✓ Removed old HF_TOKEN entry"
        else
            echo "Keeping existing entry. Exiting."
            exit 0
        fi
    fi
    
    # Add to .zshrc
    echo "" >> ~/.zshrc
    echo "# HuggingFace token for topic modeling project" >> ~/.zshrc
    echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.zshrc
    echo "✓ Added HF_TOKEN to ~/.zshrc"
    echo ""
    echo "To use it in this terminal, run: source ~/.zshrc"
    echo "Or open a new terminal window."
else
    echo ""
    echo "To use the token in this session, it's already set."
    echo "To set it manually, run: export HF_TOKEN=\"your_token_here\""
fi

echo ""
echo "=========================================="
echo "Verifying token..."
python3 -c "
import os
token = os.getenv('HF_TOKEN')
if token:
    print('✓ HF_TOKEN is set (length: {} chars)'.format(len(token)))
    print('  First 10 chars: {}...'.format(token[:10]))
else:
    print('✗ HF_TOKEN is not set')
" 2>/dev/null || echo "Could not verify token (Python check failed)"

echo ""
echo "Done! You can now test with: python3 test_chunking.py"
