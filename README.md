# Chord Detection System

A web application that detects and visualizes chord progressions in audio files using machine learning made for 7th sem major project.

## Project Structure

```
chord-detection/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   │   └── ChordPlayer.js
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   └── tailwind.config.js
├── app.py                    # Flask backend server
├── model_maker.py           # ML model creation
├── pcp_module.py           # PCP vector generation
├── running_inference.py    # Inference pipeline
├── dataset preprocessor.ipynb # Convert audio to PCP
└── requirements.txt        # Python dependencies
```

## Setup

### Backend

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask server:

```bash
python app.py
```

### Frontend

1. Navigate to frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

The application will be available at http://localhost:3000

## Features

- Real-time chord detection
- Audio playback with visualization
- Interactive timeline
- Dark theme interface
