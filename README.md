# Chord Detection System

A web application that detects and visualizes chord progressions in audio files using machine learning made for 7th sem major project.

## Project Structure

```
chord-detection/
├── frontend/                       # React frontend
│   ├── src/
│   │   ├── components/             # React components
│   │   │   └── ChordPlayer.js
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   └── tailwind.config.js
├── app.py                          # Flask backend server
├── better_onset.py                 # an improvement over the librosa beat track for no instrument audio segments
├── better_running_inference.py     # infer chords using the better onset logic with pcp model
├── model_maker.py                  # training code for pcp model
├── pcp_module.py                   # PCP vector generation
├── robust_betteronset_inference    # infer chords using the better onset logic and robust model
├── robust_model_maker.py           # training code for robust model
├── running_inference.py            # Inference pipeline
├── dataset preprocessor.ipynb      # Convert audio to PCP
└── requirements.txt                # Python dependencies
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

#### To choose between pcp and robust model comment or uncomment lines 30-33 in `app.py`

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
