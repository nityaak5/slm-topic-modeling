# HuggingFace Token Setup

## Quick Steps

1. **Get your token from HuggingFace:**
   - Go to: https://huggingface.co/settings/tokens
   - Create a new token with "Read" access
   - Copy it

2. **Set it as an environment variable:**

   **Option A: Add to ~/.zshrc (Permanent - Recommended)**
   ```bash
   echo '' >> ~/.zshrc
   echo '# HuggingFace token for topic modeling' >> ~/.zshrc
   echo 'export HF_TOKEN="your_token_here"' >> ~/.zshrc
   source ~/.zshrc
   ```
   
   Replace `your_token_here` with your actual token.

   **Option B: Set for current session only**
   ```bash
   export HF_TOKEN="your_token_here"
   ```

3. **Verify it's set:**
   ```bash
   echo $HF_TOKEN
   ```
   Should show your token (or first few characters if your shell hides it).

4. **Test it:**
   ```bash
   python3 test_chunking.py
   ```

## How it works

- `huggingface_hub` automatically reads the `HF_TOKEN` environment variable
- vLLM will use this token when downloading models
- The token is stored in your shell config, not in the code (secure)

## For HPC

When you run on HPC, you'll need to set it there too:
```bash
export HF_TOKEN="your_token_here"
```
Or add it to your HPC shell config file.
