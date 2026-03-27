from flask import Flask, request, jsonify, Response
import json
import sounddevice as sd
import threading
import time
import tempfile
import os

import audio_engine


app = Flask(__name__)
analyzer = audio_engine.AudioAnalyzer() 
#OUTPUT_CABLE_NAME = "CABLE Output" #this is for Windows 
OUTPUT_CABLE_NAME = "virtual_cable.monitor" #this is for Linux 

#INPUT_CABLE_NAME = "CABLE Input" #this is for Windows 
INPUT_CABLE_NAME = "virtual_cable" #this is for Linux  

def find_vb_cable_devices():
    """Find VB-CABLE devices"""
    devices = sd.query_devices()
    input_device = None
    output_device = None
    
    for i, dev in enumerate(devices):
        if OUTPUT_CABLE_NAME in dev['name'] and dev['max_input_channels'] > 0:
            input_device = i
        if "CABLE Input" in dev['name'] and dev['max_output_channels'] > 0:
            output_device = i
    
    return input_device, output_device

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Upload WAV, analyze, return results"""
    
    # Check file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Find devices
    input_device, output_device = find_vb_cable_devices()
    if input_device is None:
        return jsonify({'error': 'VB-CABLE Output not found'}), 500
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        file.save(tmp.name)
        wav_path = tmp.name
    
    try:
        # Reset analyzer state
        analyzer.freq_history = []
        analyzer.last_print_time = time.time()
        analyzer.start = time.time()
        analyzer.notes_sheet = []
        
        # Play audio in background
        def play_audio():
            analyzer.play_to_virtual_mic(wav_path)
        
        play_thread = threading.Thread(target=play_audio)
        play_thread.start()
        
        # Wait for playback to start
        time.sleep(0.5)
        
        # Start listening
        with sd.InputStream(
            device=input_device,
            channels=analyzer.channels,
            samplerate=analyzer.sample_rate,
            blocksize=analyzer.block_size,
            callback=analyzer.callback,
        ):
            # Wait for playback to finish + capture time
            play_thread.join()
            time.sleep(1)  # Extra capture after playback
        
        # Return results
        return Response(
                            json.dumps({
                                'success': True,
                                'notes': analyzer.notes_sheet,
                                'count': len(analyzer.notes_sheet)
                            }, indent=2, ensure_ascii=False),
                            mimetype='application/json'
                                )

    
    finally:
        # Cleanup
        os.unlink(wav_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
