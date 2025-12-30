import sys
from pathlib import Path
import streamlit as st
import requests
import io
import time

# Add the parent directory (d:\UCD\CEADAR INTERNSHIP\AA MOONSHINE) to the Python path
# This allows importing asr_ui.core.audio_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asr_backend.core import audio_utils

# FastAPI endpoint
API_ENDPOINT = "http://127.0.0.1:8000"  # Change this if your API is running elsewhere

# Page config
st.set_page_config(
    page_title="Modular ASR UI",
    page_icon="🎤",
    layout="wide"
)

# Title
st.title("🎤 Modular Automatic Speech Recognition")
st.markdown("""
This UI uses a modular ASR system. You can upload an audio file or record from your microphone.
The transcription is performed by the selected ASR model via a FastAPI backend.
""")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("ASR Settings")
    
    # Model selection (if multiple models are available)
    model_choice = st.selectbox(
        "ASR Model",
        ["Whisper JAX"],  # For now, just one model. We can extend this.
        index=0
    )
    
    # Task selection
    task = st.radio(
        "Task",
        ["Transcribe", "Translate to English"],
        index=0
    )
    task = "transcribe" if task == "Transcribe" else "translate"
    
    # Language selection
    language = st.selectbox(
        "Language (optional, leave empty for auto-detection)",
        ["", "en", "vi", "hi", "fr", "de", "es", "zh"],
        index=0
    )
    if language == "":
        language = None
        
    # Optional parameters (collapsible)
    with st.expander("Advanced Options"):
        num_beams = st.number_input("Number of beams (0 for default)", min_value=0, value=0, step=1)
        num_beams = None if num_beams == 0 else num_beams
        temperature = st.number_input("Temperature (0 for default)", min_value=0.0, value=0.0, step=0.1)
        temperature = None if temperature == 0.0 else temperature
        chunk_sec = st.number_input("Chunk seconds (0 for default)", min_value=0.0, value=0.0, step=0.5)
        chunk_sec = None if chunk_sec == 0.0 else chunk_sec
        stride_leading = st.number_input("Stride leading (0 for default)", min_value=0.0, value=0.0, step=0.1)
        stride_leading = None if stride_leading == 0.0 else stride_leading
        stride_trailing = st.number_input("Stride trailing (0 for default)", min_value=0.0, value=0.0, step=0.1)
        stride_trailing = None if stride_trailing == 0.0 else stride_trailing
        prompt = st.text_input("Prompt (optional)", value="")
        prompt = None if prompt == "" else prompt

# Main content
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Record from Microphone"])

# Helper function to call the API
def transcribe_via_api(endpoint, files=None, data=None):
    try:
        response = requests.post(endpoint, files=files, data=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response: {e.response.text}")
        return None

# Tab 1: Upload audio file
with tab1:
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac']
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        
        if st.button("Transcribe Uploaded File", key="upload_transcribe"):
            with st.spinner("Transcribing..."):
                # Prepare the data for the API
                files = {
                    'file': (uploaded_file.name, uploaded_file, uploaded_file.type)
                }
                data = {
                    'task': task,
                    'language': language or '',
                    'num_beams': num_beams or '',
                    'temperature': temperature or '',
                    'chunk_sec': chunk_sec or '',
                    'stride_leading': stride_leading or '',
                    'stride_trailing': stride_trailing or '',
                    'prompt': prompt or ''
                }
                # Remove keys with empty values
                data = {k: v for k, v in data.items() if v != ''}
                
                result = transcribe_via_api(f"{API_ENDPOINT}/transcribe/upload", files=files, data=data)
                
                if result:
                    st.success("Transcription complete!")
                    st.text_area("Transcription", result['text'], height=200)
                    
                    # Save to file
                    transcripts_dir = Path("transcripts")
                    transcripts_dir.mkdir(exist_ok=True)
                    timestamp = int(time.time())
                    save_path = transcripts_dir / f"upload_{timestamp}.txt"
                    save_path.write_text(result['text'], encoding='utf-8')
                    st.info(f"Saved to: {save_path}")

# Tab 2: Record from microphone
with tab2:
    st.header("Record from Microphone")
    
    # Recording parameters
    col1, col2 = st.columns(2)
    with col1:
        seconds = st.number_input("Recording duration (seconds)", min_value=1.0, max_value=60.0, value=8.0, step=1.0)
    with col2:
        sample_rate = st.selectbox("Sample rate", [16000, 22050, 44100], index=0)
    
    device = st.text_input("Input device (optional, leave empty for default)", "")
    if device == "":
        device = None
    
    if st.button("Start Recording", key="start_record"):
        with st.spinner(f"Recording for {seconds} seconds..."):
            try:
                # Record audio
                audio_array = audio_utils.record_audio(seconds, sample_rate, device)
                # Convert to bytes for playback
                audio_bytes = audio_utils.wav_bytes_from_array(audio_array, sample_rate)
                
                # Store in session state
                st.session_state.recorded_audio = audio_bytes
                st.session_state.recorded_sample_rate = sample_rate
                
                st.audio(audio_bytes, format="audio/wav")
                st.success("Recording complete!")
            except Exception as e:
                st.error(f"Recording failed: {e}")
                st.session_state.recorded_audio = None
    
    if 'recorded_audio' in st.session_state and st.session_state.recorded_audio is not None:
        if st.button("Transcribe Recording", key="record_transcribe"):
            with st.spinner("Transcribing..."):
                # Prepare the data for the API
                files = {
                    'file': ('recording.wav', st.session_state.recorded_audio, 'audio/wav')
                }
                data = {
                    'task': task,
                    'language': language or '',
                    'seconds': seconds,
                    'sample_rate': sample_rate,
                    'device': device or '',
                    'num_beams': num_beams or '',
                    'temperature': temperature or '',
                    'chunk_sec': chunk_sec or '',
                    'stride_leading': stride_leading or '',
                    'stride_trailing': stride_trailing or '',
                    'prompt': prompt or ''
                }
                # Remove keys with empty values
                data = {k: v for k, v in data.items() if v != ''}
                
                result = transcribe_via_api(f"{API_ENDPOINT}/transcribe/record", files=files, data=data)
                
                if result:
                    st.success("Transcription complete!")
                    st.text_area("Transcription", result['text'], height=200)
                    
                    # Save to file
                    transcripts_dir = Path("transcripts")
                    transcripts_dir.mkdir(exist_ok=True)
                    timestamp = int(time.time())
                    save_path = transcripts_dir / f"record_{timestamp}.txt"
                    save_path.write_text(result['text'], encoding='utf-8')
                    st.info(f"Saved to: {save_path}")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This application uses a modular ASR system. The backend is built with FastAPI and can be extended with additional ASR models.
""")

# Check API health
if st.sidebar.button("Check API Health"):
    try:
        response = requests.get(f"{API_ENDPOINT}/health", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("API is healthy!")
        else:
            st.sidebar.error(f"API health check failed: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"API health check failed: {e}")
