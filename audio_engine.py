import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks
import time
import threading
import soundfile as sf



# Windows-only fallback to allow quitting even if Ctrl+C isn't handled well.
'''
try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None
'''
# Audio parameters


class AudioAnalyzer():
    def __init__(self) -> None:
        self.freq_history = []
        self.last_print_time = time.time()
        self.sample_rate = 44100
        self.block_size = 8192
        self.channels = 1
        self.start = time.time()

        self.notes_sheet = []
        self.notes_lock = threading.Lock()

        # Detection parameters
        self.energy_threshold = 5e-3
        self.history_size = 5

        self.virtual_input = "CABLE Output (VB-Audio Virtual Cable)"  
        self.virtual_output = "CABLE Input (VB-Audio Virtual Cable)"   


        # Guitar frequency ranges
        self.guitar_strings = {
            'Ми 2 октава': 82.41,
            'Ля 2 октава': 110.00,
            'Ре 3 октава': 146.83,
            'Соль 3 октава': 196.00,
            'Си 3 октава': 246.94,
            'Ми 4 октава': 329.63,
        }
             
    def note_name(self, freq):
        """Convert frequency to note name - CORRECTED VERSION"""
        if freq is None:
            return "---"
        
        # A4 = 440 Hz
        notes = ['До', 'До#', 'Ре', 'Ре#', 'Ми', 'Фа', 'Фа#', 'Соль', 'Соль#', 'Ля', 'Ля#', 'Си']
        
        # Calculate semitone distance from A4
        # Formula: semitones = 12 * log2(freq / 440)
        semitones = 12 * np.log2(freq / 440.0)
        
        # Round to nearest semitone
        rounded_semitones = int(round(semitones))
        
        # Calculate note index (0 = C, 3 = E, etc.)
        # A4 is index 9 (A) at octave 4
        note_idx = (9 + rounded_semitones) % 12
        
        # Calculate octave
        octave = 4 + (9 + rounded_semitones) // 12
        
        # Handle negative octaves
        if note_idx < 0:
            note_idx += 12
            octave -= 1
        
        # Fix for E4 at 329.63 Hz
        # Semitones from A4: 12 * log2(329.63/440) = -5
        # 9 + (-5) = 4, which is E (index 4 in notes array)
        
        return f"{notes[note_idx]}{octave}"

    def cents_off(self, freq, target_freq):
        """Calculate cents deviation"""
        if freq is None or target_freq is None:
            return 0
        return 1200 * np.log2(freq / target_freq)

    def find_harmonic_peaks(self,x, fs):
        """Find all harmonic peaks"""
        x = x - np.mean(x)
        
        rms = np.sqrt(np.mean(x * x))
        if rms < self.energy_threshold:
            return None
        
        n_fft = len(x) * 4
        window = np.hanning(len(x))
        X = np.fft.rfft(x * window, n=n_fft)
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(n_fft, 1.0/fs)
        
        # Find peaks
        peaks, properties = find_peaks(mag, 
                                    height=np.max(mag) * 0.02,
                                    distance=10,
                                    prominence=np.max(mag) * 0.01)
        
        if len(peaks) == 0:
            return None
        
        peak_freqs = freqs[peaks]
        peak_mags = mag[peaks]
        
        # Sort by magnitude
        sorted_idx = np.argsort(peak_mags)[::-1]
        peak_freqs = peak_freqs[sorted_idx]
        peak_mags = peak_mags[sorted_idx]
        
        # Limit to guitar range
        valid_idx = (peak_freqs >= 70) & (peak_freqs <= 1200)
        peak_freqs = peak_freqs[valid_idx]
        peak_mags = peak_mags[valid_idx]
        
        if len(peak_freqs) == 0:
            return None
        
        return peak_freqs, peak_mags

    def weighted_harmonic_voting(self,peak_freqs, peak_mags):
        """Vote-based method using harmonic relationships"""
        votes = {}
        
        for i, freq1 in enumerate(peak_freqs[:15]):
            for j, freq2 in enumerate(peak_freqs[i+1:min(i+10, len(peak_freqs))]):
                ratio = freq2 / freq1
                
                for harmonic in range(2, 7):
                    if abs(ratio - harmonic) < 0.03:
                        fundamental = freq1
                        votes[fundamental] = votes.get(fundamental, 0) + peak_mags[i] + peak_mags[j]
                        
                        fundamental2 = freq2 / harmonic
                        if 70 < fundamental2 < 400:
                            votes[fundamental2] = votes.get(fundamental2, 0) + peak_mags[j]
        
        if not votes:
            return None
    
        return max(votes.items(), key=lambda x: x[1])[0]

    def yin_guitar(self, x, fs, min_freq=70, max_freq=400):
        """YIN algorithm for guitar range"""
        x = x.astype(np.float64)
        x = x - np.mean(x)
        
        rms = np.sqrt(np.mean(x * x))
        if rms < self.energy_threshold:
            return None
        
        tau_min = int(fs / max_freq)
        tau_max = int(fs / min_freq)
        
        if tau_max >= len(x):
            tau_max = len(x) - 1
        
        diff = np.zeros(tau_max + 1)
        for tau in range(1, tau_max + 1):
            diff[tau] = np.sum((x[tau:] - x[:-tau]) ** 2)
        
        diff_cum = np.zeros(tau_max + 1)
        diff_cum[1] = 1.0
        for tau in range(2, tau_max + 1):
            diff_cum[tau] = diff[tau] * tau / np.sum(diff[1:tau+1])
        
        best_tau = tau_min + np.argmin(diff_cum[tau_min:tau_max+1])
        pitch = fs / best_tau
        
        if min_freq <= pitch <= max_freq:
            return pitch
        
        return None

    def find_fundamental_from_harmonics(self, peak_freqs, peak_mags):
        """Identify fundamental from harmonic series"""
        best_fundamental = None
        best_score = 0
        
        for i, test_fund in enumerate(peak_freqs[:10]):
            if test_fund < 70 or test_fund > 400:
                continue
            
            score = 0
            harmonics_found = 0
            
            for harmonic in [2, 3, 4, 5, 6]:
                harmonic_freq = test_fund * harmonic
                if len(peak_freqs) > 0:
                    closest_idx = np.argmin(np.abs(peak_freqs - harmonic_freq))
                    freq_error = abs(peak_freqs[closest_idx] - harmonic_freq) / harmonic_freq
                    
                    if freq_error < 0.03:
                        score += peak_mags[closest_idx] / harmonic
                        harmonics_found += 1
            
            score += peak_mags[i]
            score *= (1 + harmonics_found / 3)
            
            if score > best_score:
                best_score = score
                best_fundamental = test_fund
        
        return best_fundamental

    def detect_guitar_pitch(self, x, fs):
        """Combined approach for guitar"""
        peaks = self.find_harmonic_peaks(x, fs)
        if peaks is None:
            return None
        
        peak_freqs, peak_mags = peaks
        
        estimates = []
        
        fundamental1 = self.weighted_harmonic_voting(peak_freqs, peak_mags)
        if fundamental1:
            estimates.append(fundamental1)
        
        fundamental2 = self.yin_guitar(x, fs)
        if fundamental2:
            estimates.append(fundamental2)
        
        fundamental3 = self.find_fundamental_from_harmonics(peak_freqs, peak_mags)
        if fundamental3:
            estimates.append(fundamental3)
        
        if not estimates:
            return None
        
        return np.median(estimates)

    def play_to_virtual_mic(self, wav_file):
        """Play WAV file to CABLE Input"""
        data, fs = sf.read(wav_file)
        
        # Find CABLE Input device
        devices = sd.query_devices()
        out_device = None
        for i, dev in enumerate(devices):
            if "CABLE Input" in dev['name']:
                out_device = i
                print(f"Playing to: {dev['name']}")
                break
        
        if out_device is None:
            print("CABLE Input not found!")
            return
        
        # Play the file
        sd.play(data, fs, device=out_device)
        sd.wait()
        print("Playback finished")

    # Test the note name function
    def test_note_name_function(self):
        print("Testing note name function:")
        test_freqs = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
        for f in test_freqs:
            print(f"{f:.2f} Hz -> {self.note_name(f)}")
        print()

        # Global variables
        #freq_history = []
        #last_print_time = time.time()

    def callback(self,indata, frames, t, status):
        
        if status:
            print("Status:", status)
        
        x = indata[:, 0].astype(np.float64)
        hz = self.detect_guitar_pitch(x, self.sample_rate)
        
        if hz:
            self.freq_history.append(hz)
            if len(self.freq_history) > self.history_size:
                self.freq_history.pop(0)
            hz_smoothed = np.median(self.freq_history)
        else:
            self.freq_history = []
            hz_smoothed = None
        
        current_time = time.time()
        time_elapsed =current_time - self.start 
        if current_time - self.last_print_time >= 0.1:
            if hz_smoothed:
                note = self.note_name(hz_smoothed)
                
                # Find closest string
                best_match = None
                best_error = float('inf')
                for string_name, string_freq in self.guitar_strings.items():
                    error = abs(hz_smoothed - string_freq)
                    if error < best_error:
                        best_error = error
                        best_match = (string_name, string_freq)
                
                if best_match and best_error < 15:
                    string_name, string_freq = best_match
                    cents = self.cents_off(hz_smoothed, string_freq)
                    status_str = f"{cents:+.0f} cents"
                    if abs(cents) < 3:
                        status_str = "✓ Perfect!"
                    elif abs(cents) < 10:
                        status_str += " (good)"
                    print(f"{hz_smoothed:.1f} Hz  |  {note:4}  |  {status_str} |  {time_elapsed}")
                    with self.notes_lock:
                        self.notes_sheet.append({
                        'frequency': round(hz_smoothed, 1),
                        'note': note.strip(),
                        'status': status_str,
                        'time': time_elapsed
                        })

                        #self.notes_sheet.append([
                        #    f"{hz_smoothed:.1f} Hz  |  {note:4}   |  {status_str}  |  {time_elapsed}"
                        #])

                else:
                    print(f"{hz_smoothed:.1f} Hz  |  {note:4}  |  ---  |  Detected |  {time_elapsed}")
                    with self.notes_lock:
                        status_str = "Detected"
                        self.notes_sheet.append({
                        'frequency': round(hz_smoothed, 1),
                        'note': note.strip(),
                        'status': status_str,
                        'time': time_elapsed
                        })
                        #self.notes_sheet.append([
                        #    f"{hz_smoothed:.1f} Hz  |  {note:4}  |  ---  |  Detected  |  {time_elapsed}"
                        #])
            else:
                print("🎸 [silence]")
            
            self.last_print_time = current_time

'''
try: 
    print("Available devices:")
    print(sd.query_devices())
    
    # Find VB-CABLE devices
    devices = sd.query_devices()
    input_device = None
    output_device = None
    
    for i, dev in enumerate(devices):
        if "CABLE Output" in dev['name'] and dev['max_input_channels'] > 0:
            input_device = i
            print(f"✓ Found input device {i}: {dev['name']}")
        if "CABLE Input" in dev['name'] and dev['max_output_channels'] > 0:
            output_device = i
            print(f"✓ Found output device {i}: {dev['name']}")
    
    if input_device is None:
        print("❌ CABLE Output not found. Make sure VB-CABLE is installed.")
        exit(1)

    play_thread = threading.Thread(
        target=analyzer.play_to_virtual_mic, 
        args=(r"C:\python projects\vocal_coach\Meladze_Vocals.wav",)
    )
    play_thread.start()
    
    # Wait a moment for playback to start
    time.sleep(0.5)


    with sd.InputStream(
        device = input_device,
        channels=CHANNELS, 
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        print("✅ Ready! Play a string...\n")
        START = time.time()

        while True:
            # Keep the main thread responsive so Ctrl+C works reliably.
            sd.sleep(50)
            if msvcrt and msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    ch = ch.decode("utf-8", errors="ignore").strip().lower()
                except Exception:
                    ch = ""
                if ch == "q":
                    raise KeyboardInterrupt
except KeyboardInterrupt:

    print("\n✅ Stopped.")
    with notes_lock:
        items = list(notes_sheet)
    with open("output.txt", "w", encoding="utf-8") as f:
        for item in items:
            f.write(str(item) + "\n")

curl.exe -X POST -F "file=@C:\python projects\vocal_coach\Meladze_Vocals.wav" http://localhost:5000/analyze -o "C:\python projects\vocal_coach\result.txt"
'''


