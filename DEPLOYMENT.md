# Plant Disease Detector - Deployment Guide

## Deployment Options

### Option 1: Render (Recommended - Free & Easy)

**Pros:** Free tier, automatic HTTPS, easy setup, supports ML models
**Model Size Limit:** 512MB (your model should fit)

**Steps:**
1. Push your code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/plant-disease-detector.git
   git push -u origin main
   ```

2. Go to [render.com](https://render.com) and sign up
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn web_interface:app`
   - **Environment:** Python 3
6. Click "Create Web Service"
7. Your app will be live at: `https://your-app-name.onrender.com`

**Note:** Free tier spins down after inactivity (slow first load)

---

### Option 2: Railway

**Pros:** Easy deployment, generous free tier, fast performance

**Steps:**
1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. "New Project" → "Deploy from GitHub"
4. Select your repository
5. Railway auto-detects and deploys your Flask app
6. Get your deployment URL

---

### Option 3: Hugging Face Spaces

**Pros:** Free, designed for ML models, good for showcasing

**Steps:**
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space → "Gradio" or "Streamlit" 
3. Upload your files via web interface or git
4. Add `app.py` that wraps your Flask app in Gradio

**Note:** Consider converting to Gradio for better Hugging Face integration

---

### Option 4: Google Cloud Run

**Pros:** Scalable, pay-per-use, free tier available

**Requirements:**
- Google Cloud account with billing enabled
- Docker installed

**Steps:**

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8080
   CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "web_interface:app"]
   ```

2. Deploy:
   ```bash
   gcloud run deploy plant-disease-detector \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

---

### Option 5: AWS EC2 (Advanced)

**Pros:** Full control, scalable
**Cons:** Requires manual setup, costs start immediately

**Steps:**
1. Launch EC2 instance (t2.micro for free tier)
2. SSH into instance
3. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip nginx -y
   ```
4. Clone your repository
5. Install packages: `pip3 install -r requirements.txt`
6. Configure Nginx as reverse proxy
7. Use systemd to run app as service

---

### Option 6: Heroku (Classic Option)

**Note:** No longer has free tier, but very reliable

**Steps:**
1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create plant-disease-detector`
4. Deploy:
   ```bash
   git push heroku main
   ```

---

## Important Considerations

### Model Size Warning
Your `.keras` model file (~30-50MB typically) might cause issues:

**Solutions:**
1. **GitHub LFS** (Large File Storage):
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   git add models/detector_disease.keras
   git commit -m "Add model with LFS"
   ```

2. **External Storage** (Recommended for large models):
   - Upload model to Google Drive, Dropbox, or AWS S3
   - Download at runtime:
   ```python
   import gdown
   url = 'YOUR_GOOGLE_DRIVE_SHARE_LINK'
   output = 'models/detector_disease.keras'
   if not os.path.exists(output):
       gdown.download(url, output, quiet=False)
   ```

3. **Model Compression**: Use TensorFlow Lite for smaller model

### Security Improvements
Add to `web_interface.py`:
```python
from flask_cors import CORS
CORS(app)  # Enable CORS if needed

# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response
```

### Environment Variables
For sensitive config, use environment variables:
```python
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SECRET_KEY'] = SECRET_KEY
```

### Database for Feedback
Consider adding PostgreSQL for feedback logs:
- Render: Built-in PostgreSQL
- Heroku: Heroku Postgres addon
- Railway: PostgreSQL plugin

---

## Quick Start (Render - Fastest)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   gh repo create plant-disease-detector --public
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Visit render.com
   - New Web Service → Connect GitHub repo
   - Auto-deploys! ✓

3. **Access your live app:**
   `https://plant-disease-detector.onrender.com`

---

## Testing Before Deployment

Run production server locally:
```bash
gunicorn web_interface:app --bind 0.0.0.0:5000
```

---

## Monitoring & Maintenance

- **Logs:** Check platform dashboard for errors
- **Uptime:** Use UptimeRobot for monitoring
- **Analytics:** Add Google Analytics to templates
- **Updates:** Push to GitHub, auto-deploys

---

## Cost Estimates (as of 2026)

- **Render:** Free tier (enough for demo/portfolio)
- **Railway:** $5/month after free credits
- **Hugging Face:** Free
- **Google Cloud Run:** ~$5-10/month (1000 requests/day)
- **AWS EC2:** ~$10-15/month (t2.micro)
- **Heroku:** ~$7/month (Eco dyno)

---

## Need Help?

Common issues:
- **Model not found:** Check path in `web_interface.py`
- **Out of memory:** Use smaller batch size or TF Lite
- **Slow loading:** Enable caching, use CDN for static files
- **Upload errors:** Check `UPLOAD_FOLDER` permissions

For questions, check platform documentation or open GitHub issue.
