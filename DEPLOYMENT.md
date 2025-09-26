# 🚀 AI Chat Pro - Deployment Guide

## Quick Deployment to Streamlit Cloud

### Step 1: Prepare Your Repository
1. **Commit all files to GitHub:**
   ```bash
   git add .
   git commit -m "AI Chat Pro - Ready for deployment"
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. **Go to:** https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in:**
   - **Repository:** `YOUR_USERNAME/YOUR_REPO_NAME`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. **Click "Deploy!"**

### Step 3: Configure Environment Variables
After deployment, add your API keys:

1. **Go to your deployed app**
2. **Click hamburger menu (☰) → Settings → Secrets**
3. **Add this configuration:**
   ```toml
   OPENAI_API_KEY = "your_openai_key_here"
   ANTHROPIC_API_KEY = "your_anthropic_key_here"
   GOOGLE_API_KEY = "your_google_key_here"
   GROQ_API_KEY = "your_groq_key_here"
   LANGCHAIN_API_KEY = "your_langsmith_key_here"
   SESSION_TAG = "production"
   ```
4. **Click "Save"**

### Step 4: Access Your App
Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`

## Required Files for Deployment

### ✅ Core Files (Must Include):
- `app.py` - Entry point
- `main_ui.py` - Main application
- `main9.py` - LangChain integration
- `requirements.txt` - Dependencies
- `.streamlit/config.toml` - Streamlit config
- `.gitignore` - Exclude unnecessary files

### ❌ Files to Exclude (via .gitignore):
- `conversations/` - Local conversation data
- `logs/` - Local log files
- `__pycache__/` - Python cache
- `.env` - Environment variables (use Streamlit secrets instead)
- `poetry.lock` - Poetry lock file

## Alternative Deployment Options

### Railway
- Go to: https://railway.app/
- Connect GitHub repository
- Add environment variables
- Deploy automatically

### Heroku
- Go to: https://heroku.com/
- Create new app
- Connect GitHub repository
- Add environment variables in Settings
- Deploy

## Troubleshooting

### Common Issues:
1. **Import errors:** Check `requirements.txt` has all dependencies
2. **API key errors:** Verify environment variables are set correctly
3. **Port issues:** Ensure app uses `$PORT` environment variable
4. **Memory issues:** Consider upgrading to paid tier for larger apps

### Support:
- Check Streamlit Cloud logs for detailed error messages
- Verify all API keys are valid and have sufficient credits
- Ensure your repository is public (for free Streamlit Cloud)
