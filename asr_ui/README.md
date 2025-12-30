# Modular ASR UI

A modular Automatic Speech Recognition (ASR) system with interchangeable models, FastAPI backend, and Streamlit frontend.

## Project Structure

```
asr_ui/
├── __init__.py
├── README.md
├── core/
│   └── audio_utils.py          # Audio recording and conversion utilities
├── models/
│   ├── __init__.py             # Model factory and registry
│   ├── base.py                 # Abstract base class for ASR models
│   └── whisper_jax.py          # Whisper JAX model implementation
├── api/
│   └── main.py                 # FastAPI backend with REST endpoints
└── ui/
    └── app.py                  # Streamlit frontend application
```

## Features

- **Modular Design**: Easily add new ASR models by implementing the `ASRModel` abstract class
- **FastAPI Backend**: RESTful API for transcription with support for file upload and microphone recording
- **Streamlit Frontend**: User-friendly web interface for testing and demonstration
- **Extensible**: Designed to support multiple ASR models (Whisper JAX, OmniLingual, etc.)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd asr_ui
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the System

### Option 1: Run Everything Together (Recommended for Development)

Use the provided runner script:

```bash
python run.py
```

This will:
1. Start the FastAPI backend on http://127.0.0.1:8000
2. Start the Streamlit frontend on http://127.0.0.1:8501

### Option 2: Run Components Separately

#### Start the FastAPI Backend:

```bash
cd asr_ui
python -m api.main
```

The API will be available at http://127.0.0.1:8000

#### Start the Streamlit Frontend:

In a new terminal:

```bash
cd asr_ui
streamlit run ui/app.py
```

The UI will be available at http://127.0.0.1:8501

## API Endpoints

### GET /
Health check and basic information.

### GET /health
Health check for the ASR model.

### GET /models
List available models and current model information.

### POST /transcribe/upload
Transcribe an uploaded audio file.

**Parameters:**
- `file`: Audio file (multipart/form-data)
- `task`: "transcribe" or "translate" (default: "transcribe")
- `language`: Language code (optional)
- Additional Whisper parameters: `num_beams`, `temperature`, `chunk_sec`, etc.

### POST /transcribe/record
Record from microphone and transcribe.

**Parameters:**
- `seconds`: Recording duration (default: 8.0)
- `sample_rate`: Sample rate (default: 16000)
- `device`: Audio device (optional)
- Other parameters same as upload endpoint

## Adding a New ASR Model

1. Create a new model class in `models/` that inherits from `ASRModel`:
```python
# models/new_model.py
from .base import ASRModel

class NewASRModel(ASRModel):
    def __init__(self, **kwargs):
        # Initialize your model
        pass
    
    def transcribe(self, audio_bytes, task="transcribe", language=None, **kwargs):
        # Implement transcription logic
        return "transcribed text"
    
    def get_available_languages(self):
        return ["en", "es", "fr"]  # List supported languages
    
    def get_model_info(self):
        return {"name": "New Model", "version": "1.0"}
```

2. Register the model in `models/__init__.py`:
```python
from .new_model import NewASRModel

MODEL_REGISTRY = {
    "whisper_jax": WhisperJAXModel,
    "new_model": NewASRModel,  # Add this line
}
```

3. The model will now be available in the API and UI.

## Configuration

Environment variables:

- `ASR_DEFAULT_MODEL`: Default model to use (default: "whisper_jax")
- `WHISPER_ENDPOINT`: Whisper JAX API endpoint (default: "http://127.0.0.1:8008/transcribe")
- `API_HOST`: FastAPI host (default: "0.0.0.0")
- `API_PORT`: FastAPI port (default: 8000)

## Deployment

### Deploying to Contabo Server

1. **Set up the server**:
```bash
# Install Python and dependencies
sudo apt update
sudo apt install python3-pip python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
# Set environment variables
export WHISPER_ENDPOINT="http://127.0.0.1:8008/transcribe"
export ASR_DEFAULT_MODEL="whisper_jax"
```

3. **Run with PM2 (for process management)**:
```bash
# Install PM2
npm install -g pm2

# Start the API
pm2 start "python -m asr_ui.api.main" --name "asr-api"

# Start the UI
pm2 start "streamlit run asr_ui/ui/app.py --server.port 8501 --server.address 0.0.0.0" --name "asr-ui"

# Save PM2 configuration
pm2 save
pm2 startup
```

4. **Set up Nginx as reverse proxy** (optional):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

1. **API connection errors**:
   - Ensure the FastAPI server is running
   - Check the `API_ENDPOINT` in `ui/app.py`
   - Verify CORS settings in `api/main.py`

2. **Audio recording issues**:
   - Check microphone permissions
   - Verify sounddevice is installed correctly
   - Test with `python -c "import sounddevice as sd; print(sd.query_devices())"`

3. **Model loading errors**:
   - Verify the Whisper JAX server is running
   - Check the `WHISPER_ENDPOINT` environment variable

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
