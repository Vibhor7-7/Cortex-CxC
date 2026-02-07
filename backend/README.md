# Cortex Backend Setup

## Environment Configuration

This project uses environment variables for configuration. Follow these steps:

### For New Contributors:

1. **Copy the example environment file:**
   ```bash
   cd backend
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys:**
   ```bash
   # On macOS/Linux
   nano .env
   # or
   code .env
   ```

3. **Get your OpenAI API key:**
   - Visit https://platform.openai.com/api-keys
   - Create a new API key
   - Paste it into `.env` replacing `your_openai_api_key_here`

4. **Verify your setup:**
   ```bash
   # Show hidden files to see .env (macOS Finder)
   # Press: Cmd + Shift + .

   # Or list in terminal
   ls -la
   ```

### Security Notes

-  `.env` is already in `.gitignore` - it will NOT be committed
-  Each contributor maintains their own local `.env` file
-  Never share your `.env` file or commit it to version control
-  Use `.env.example` as the template (this IS committed)

### Troubleshooting

**Can't see `.env` file?**
- Files starting with `.` are hidden by default on macOS/Linux
- In Finder: Press `Cmd + Shift + .` to show hidden files
- In terminal: Use `ls -la` instead of `ls`

**`.env` file doesn't exist?**
- Run `cp .env.example .env` from the `backend/` directory
- The file should be created at `backend/.env`

**API key not working?**
- Verify you copied the entire key (starts with `sk-proj-`)
- Check for extra spaces or newlines
- Generate a new key at https://platform.openai.com/api-keys
