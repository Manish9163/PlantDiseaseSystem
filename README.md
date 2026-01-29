# Plant Disease Detection System

AI-powered plant disease detection using deep learning. Upload plant images to get instant disease diagnosis and treatment recommendations.

## Features
- 38 plant disease classifications
- Real-time image processing
- Treatment recommendations
- User feedback system
- Mobile-friendly interface

## Technologies
- Flask web framework
- TensorFlow/Keras for ML
- MobileNetV2 architecture
- OpenCV for image processing

## Deployment
This application is deployed on Render. See [DEPLOYMENT.md](DEPLOYMENT.md) for details.

## Local Development
1. Clone repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python web_interface.py`
6. Visit: `http://localhost:5000`

## Model
- Pre-trained MobileNetV2 on PlantVillage dataset
- 38 disease classes
- 240x240 input size
- ~30MB model file

## License
Educational/Research purposes
