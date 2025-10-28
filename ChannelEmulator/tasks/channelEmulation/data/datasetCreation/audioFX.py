# audioFX.py
# Audio Effects Processor - Applies various effects to audio signals

import argparse
import numpy as np
import os
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# ==========================================================
# AUDIO I/O FUNCTIONS
# ==========================================================
def load_audio(filename, sample_rate=None):
    """
    Load audio from either WAV or TXT format.
    
    Args:
        filename: Path to audio file (.wav or .txt)
        sample_rate: Sample rate for TXT files (required for TXT)
    
    Returns:
        audio_data: Audio signal as numpy array
        fs: Sample rate
    """
    if filename.lower().endswith('.wav'):
        audio_data, fs = sf.read(filename, always_2d=False)
        if audio_data.ndim > 1:
            if audio_data.shape[1] != 1:
                print(f"Loaded WAV: {filename} with {audio_data.shape[1]} channels; downmixing to mono.")
            audio_data = np.mean(audio_data, axis=1)
        print(f"Loaded WAV: {filename} (fs={fs} Hz, samples={len(audio_data)})")
        return audio_data.astype(np.float32), fs
    
    elif filename.lower().endswith('.txt'):
    # Read header
        fs = None
        bits = None
        samples = None
        date = None
        header_lines = []
        with open(filename, 'r', encoding='utf-8') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith('#'):
                    header_lines.append(line.strip())
                else:
                    f.seek(pos)
                    break
            # Leer datos cuantizados ignorando cabecera
            quantized_data = np.loadtxt(f, dtype=np.int64)
    # Extract metadata
        for h in header_lines:
            if h.startswith('# fs='):
                fs = int(h.split('=')[1])
            elif h.startswith('# bits='):
                bits = int(h.split('=')[1])
            elif h.startswith('# samples='):
                samples = int(h.split('=')[1])
            elif h.startswith('# date='):
                date = h.split('=')[1]
        # If no metadata, use arguments
        if fs is None:
            if sample_rate is not None:
                fs = sample_rate
                print(f"WARNING: 'fs' not found in header, using argument --fs={fs}")
            else:
                print(f"ERROR: Could not read sample rate (fs) from TXT header '{filename}' and no argument was provided.")
                raise ValueError("Cannot continue: missing sample rate in TXT and argument.")
        if bits is None:
            bits = 12  # Default value if not found in header or argument
            print(f"WARNING: 'bits' not found in header, using default bits={bits}")
        max_val = 2**bits - 1
        audio_data = (quantized_data.astype(np.float32) / max_val) * 2.0 - 1.0
        print(f"Loaded TXT: {filename} (fs={fs} Hz, samples={len(audio_data)}, bits={bits}, date={date})")
        return audio_data, fs
    
    else:
        raise ValueError("Unsupported file format. Use .wav or .txt")


def save_audio(audio_data, filename, fs, bits=None):
    """
    Save audio to either WAV or TXT format.
    
    Args:
        audio_data: Audio signal as numpy array
        filename: Output filename (.wav or .txt)
        fs: Sample rate
        bits: Bit depth for TXT quantization (default: 12)
    """
    if filename.lower().endswith('.wav'):
        sf.write(filename, audio_data.astype(np.float32), fs, subtype='PCM_16')
        print(f"Saved WAV: {filename}")
    
    elif filename.lower().endswith('.txt'):
        if bits is None:
            bits = 12
        
        # Clamp and quantize
        audio_clamped = np.clip(audio_data, -1.0, 1.0)
        quantized = np.rint((audio_clamped + 1.0) / 2.0 * (2**bits - 1)).astype(np.int64)
        quantized = np.clip(quantized, 0, 2**bits - 1)
        
        np.savetxt(filename, quantized, fmt='%d', delimiter='\t')
        print(f"Saved TXT: {filename} (bits={bits})")
    
    else:
        raise ValueError("Unsupported output format. Use .wav or .txt")


# ==========================================================
# AUDIO EFFECTS
# ==========================================================
def apply_delay(audio, fs, delay_time=0.3, feedback=0.4, mix=0.5, output_gain=1.0):
    """
    Apply delay effect.
    
    Args:
        audio: Input audio signal
        fs: Sample rate
        delay_time: Delay time in seconds
        feedback: Feedback amount (0-0.95)
        mix: Wet/dry mix (0=dry, 1=wet)
    """
    delay_samples = int(delay_time * fs)
    if delay_samples >= len(audio):
        delay_samples = len(audio) - 1
    
    output = np.copy(audio)
    
    for i in range(delay_samples, len(audio)):
        delayed_sample = output[i - delay_samples] * feedback
        output[i] = audio[i] + delayed_sample
    
    # Apply wet/dry mix
    return (audio * (1 - mix) + output * mix) * output_gain


def apply_clipping(audio, threshold=0.7):
    """
    Apply hard clipping distortion.
    
    Args:
        audio: Input audio signal
        threshold: Clipping threshold (0-1)
    """
    return np.clip(audio, -threshold, threshold)


def apply_saturation(audio, drive=2.0, mix=1.0, output_gain=1.0):
    """
    Apply soft saturation/overdrive.
    
    Args:
        audio: Input audio signal
        drive: Drive amount (>1 for saturation)
        mix: Wet/dry mix (0=dry, 1=wet)
    """
    driven = audio * drive
    saturated = np.tanh(driven)
    
    return (audio * (1 - mix) + saturated * mix) * output_gain


def apply_compression(audio, threshold=0.5, ratio=4.0, attack=0.01, release=0.1, fs=48000, output_gain=1.0):
    """
    Apply dynamic range compression.
    
    Args:
        audio: Input audio signal
        threshold: Compression threshold (0-1)
        ratio: Compression ratio
        attack: Attack time in seconds
        release: Release time in seconds
        fs: Sample rate
    """
    # Simple envelope follower and gain reduction
    envelope = np.abs(audio)
    
    # Smooth envelope
    attack_coeff = np.exp(-1.0 / (attack * fs))
    release_coeff = np.exp(-1.0 / (release * fs))
    
    smoothed_env = np.zeros_like(envelope)
    smoothed_env[0] = envelope[0]
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed_env[i-1]:
            smoothed_env[i] = envelope[i] * (1 - attack_coeff) + smoothed_env[i-1] * attack_coeff
        else:
            smoothed_env[i] = envelope[i] * (1 - release_coeff) + smoothed_env[i-1] * release_coeff
    
    # Calculate gain reduction
    gain_reduction = np.ones_like(smoothed_env)
    over_threshold = smoothed_env > threshold
    gain_reduction[over_threshold] = threshold + (smoothed_env[over_threshold] - threshold) / ratio
    gain_reduction[over_threshold] /= smoothed_env[over_threshold]
    
    return audio * gain_reduction * output_gain


def apply_distortion(
    audio,
    drive=5.0,
    level=1.0,
    mix=1.0,
    distortion_type='tube',
    fs=48000,
    post_lp_cutoff=12000.0,
    output_gain=1.0,
):
    """
    Apply musical distortion pedal effect with multiple distortion types.
    
    Args:
        audio: Input audio signal
        drive: Amount of distortion/overdrive (1.0-20.0, typical range)
        level: Output level compensation (0.1-2.0)
        mix: Wet/dry mix (0=dry, 1=wet)
        distortion_type: Type of distortion ('tube', 'fuzz', 'overdrive', 'clipping')
        fs: Sample rate for post-filtering
        post_lp_cutoff: Post-EQ low-pass cutoff frequency in Hz (default 12 kHz)
        output_gain: Additional output gain scaling
    
    Returns:
        Processed audio signal
    """
    # Input gain staging - simulate input buffer/preamp
    driven_signal = audio * drive
    
    if distortion_type == 'tube':
        # Tube amplifier saturation - asymmetric soft clipping
        # Simulates vacuum tube characteristics
        positive = driven_signal >= 0
        negative = driven_signal < 0
        
        distorted = np.zeros_like(driven_signal)
        # Asymmetric response (tubes clip positive/negative differently)
        distorted[positive] = np.tanh(driven_signal[positive] * 0.7) * 1.2
        distorted[negative] = np.tanh(driven_signal[negative] * 0.8) * 0.9
        
        # Add subtle even harmonics (tube characteristic)
        distorted += 0.05 * np.tanh(driven_signal * 2.0) * driven_signal
        
    elif distortion_type == 'fuzz':
        # Fuzz box - hard clipping with compression
        # Simulates transistor fuzz pedals
        compressed = np.tanh(driven_signal * 0.5)  # Pre-compression
        threshold = 0.7
        distorted = np.clip(compressed * 3.0, -threshold, threshold)
        
        # Add octave-up harmonics (fuzz characteristic)
        octave_up = np.clip(driven_signal * 10.0, -0.3, 0.3)
        distorted += 0.1 * octave_up
        
    elif distortion_type == 'overdrive':
        # Overdrive pedal - smooth tube-like saturation
        # Simulates Ibanez Tube Screamer style
        # Mid-frequency emphasis before distortion
        emphasis_freq = 500  # Hz - typical mid boost frequency
        nyquist = 22050  # Assume 44.1kHz sample rate
        normalized_freq = emphasis_freq / nyquist
        
        # Simple high-pass + low-pass for mid emphasis
        if normalized_freq < 0.5:
            b_hp, a_hp = signal.butter(1, normalized_freq * 0.3, 'high')
            b_lp, a_lp = signal.butter(1, normalized_freq * 3.0, 'low')
            emphasized = signal.filtfilt(b_lp, a_lp, 
                        signal.filtfilt(b_hp, a_hp, driven_signal))
        else:
            emphasized = driven_signal
        
        # Smooth tube-like saturation
        distorted = np.tanh(emphasized * 0.8)
        
        # Soft knee compression
        envelope = np.abs(distorted)
        compressed_env = np.tanh(envelope * 1.5) / 1.5
        distorted = distorted * (compressed_env / (envelope + 1e-8))
        
    elif distortion_type == 'clipping':
        # Diode clipping - classic distortion pedal
        # Simulates diode clipping circuits
        threshold_pos = 0.7
        threshold_neg = -0.6  # Asymmetric clipping
        
        # Soft clipping with diode curve simulation
        distorted = np.copy(driven_signal)
        
        # Positive clipping
        pos_mask = distorted > threshold_pos
        distorted[pos_mask] = threshold_pos + np.tanh((distorted[pos_mask] - threshold_pos) * 3.0) * 0.3
        
        # Negative clipping  
        neg_mask = distorted < threshold_neg
        distorted[neg_mask] = threshold_neg + np.tanh((distorted[neg_mask] - threshold_neg) * 3.0) * 0.3
        
    else:
        # Default: simple waveshaping
        distorted = np.tanh(driven_signal)
    
    # Post-distortion low-pass to tame aliasing/harshness
    if fs and fs > 0 and post_lp_cutoff:
        nyquist = fs / 2.0
        cutoff = np.clip(post_lp_cutoff, 1000.0, nyquist * 0.95)
        norm_cutoff = cutoff / nyquist
        b_lp, a_lp = signal.butter(2, norm_cutoff, btype="lowpass")
        distorted = signal.lfilter(b_lp, a_lp, distorted)
    
    # Level compensation - adjust output volume
    # Distortion typically reduces peak amplitude but increases RMS
    level_compensated = distorted * level
    
    # Soft limiting to prevent excessive peaks
    level_compensated = np.tanh(level_compensated * 0.95) / 0.95
    
    # Final wet/dry mix
    return (audio * (1 - mix) + level_compensated * mix) * output_gain


def get_distortion_presets():
    """
    Get dictionary of distortion pedal presets.
    
    Returns:
        Dictionary with preset names and parameters
    """
    presets = {
        'clean_boost': {'drive': 1.2, 'level': 1.5, 'mix': 1.0, 'type': 'tube'},
        'light_overdrive': {'drive': 3.0, 'level': 1.3, 'mix': 1.0, 'type': 'overdrive'},
        'classic_overdrive': {'drive': 5.0, 'level': 1.2, 'mix': 1.0, 'type': 'overdrive'},
        'tube_distortion': {'drive': 7.0, 'level': 1.0, 'mix': 1.0, 'type': 'tube'},
        'heavy_distortion': {'drive': 10.0, 'level': 0.8, 'mix': 1.0, 'type': 'clipping'},
        'vintage_fuzz': {'drive': 8.0, 'level': 0.9, 'mix': 1.0, 'type': 'fuzz'},
        'modern_fuzz': {'drive': 12.0, 'level': 0.7, 'mix': 1.0, 'type': 'fuzz'},
        'light_saturation': {'drive': 2.0, 'level': 1.4, 'mix': 0.7, 'type': 'tube'},
    }
    return presets


def apply_distortion_preset(audio, preset_name, fs=48000, post_lp_cutoff=12000.0, output_gain=1.0):
    """
    Apply distortion using a preset.
    
    Args:
        audio: Input audio signal
        preset_name: Name of the preset to apply
    
    Returns:
        Processed audio signal
    """
    presets = get_distortion_presets()
    if preset_name not in presets:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {list(presets.keys())}")
        return audio
    
    params = presets[preset_name]
    return apply_distortion(
        audio,
        drive=params['drive'],
        level=params['level'],
        mix=params['mix'],
        distortion_type=params['type'],
        fs=fs,
        post_lp_cutoff=post_lp_cutoff,
        output_gain=output_gain,
    )


def apply_noise(audio, noise_level=0.05, noise_type='white', output_gain=1.0):
    """
    Add noise to the signal.
    
    Args:
        audio: Input audio signal
        noise_level: Noise amplitude
        noise_type: 'white', 'pink', or 'brown'
    """
    if noise_type == 'white':
        noise = np.random.normal(0, noise_level, len(audio))
    elif noise_type == 'pink':
        # Approximate pink noise using filtered white noise
        white_noise = np.random.normal(0, 1, len(audio))
        # Simple pink noise filter (1/f characteristic)
        b, a = signal.butter(1, 0.1, 'low')
        noise = signal.filtfilt(b, a, white_noise)
        noise -= np.mean(noise)
        noise /= np.max(np.abs(noise)) + 1e-8
        noise *= noise_level
    elif noise_type == 'brown':
        # Brown noise (Brownian/red noise)
        white_noise = np.random.normal(0, 1, len(audio))
        noise = np.cumsum(white_noise)
        noise -= np.mean(noise)
        noise /= np.max(np.abs(noise)) + 1e-8
        noise *= noise_level
    else:
        noise = np.random.normal(0, 1, len(audio))
        noise -= np.mean(noise)
        noise /= np.max(np.abs(noise)) + 1e-8
        noise *= noise_level
    
    return (audio + noise) * output_gain


def apply_echo(audio, fs, delay_time=0.5, decay=0.3, num_echoes=3, output_gain=1.0):
    """
    Apply echo effect with multiple reflections.
    
    Args:
        audio: Input audio signal
        fs: Sample rate
        delay_time: Time between echoes in seconds
        decay: Amplitude decay per echo
        num_echoes: Number of echo repetitions
    """
    delay_samples = int(delay_time * fs)
    output = np.copy(audio)
    
    for echo in range(1, num_echoes + 1):
        echo_delay = delay_samples * echo
        echo_gain = decay ** echo
        
        if echo_delay < len(audio):
            # Add echoes that fit within the signal length
            output[echo_delay:] += audio[:-echo_delay] * echo_gain
        else:
            # Extend signal for longer echoes
            extended_length = len(audio) + echo_delay
            extended_output = np.zeros(extended_length)
            extended_output[:len(output)] = output
            extended_output[echo_delay:echo_delay+len(audio)] += audio * echo_gain
            output = extended_output
    
    return output * output_gain


def apply_doppler_effect(audio, fs, speed_profile=None, max_shift=0.1, output_gain=1.0):
    """
    Apply Doppler effect simulation.
    
    Args:
        audio: Input audio signal
        fs: Sample rate
        speed_profile: Array of relative velocities (-1 to 1), or None for sine wave
        max_shift: Maximum frequency shift ratio
    """
    if speed_profile is None:
        # Create a sine wave speed profile
        t = np.linspace(0, 2*np.pi, len(audio))
        speed_profile = np.sin(t) * 0.5  # -0.5 to 0.5
    
    # Resample with time-varying sample rate to simulate Doppler
    time_original = np.arange(len(audio)) / fs
    
    # Calculate time warping based on speed profile
    time_stretch = 1 + speed_profile * max_shift
    time_warped = np.cumsum(time_stretch) / fs
    time_warped -= time_warped[0]
    max_time = time_original[-1]
    time_warped = np.clip(time_warped, 0.0, max_time)
    
    # Interpolate to create Doppler effect
    interp_func = interp1d(time_original, audio, kind='linear', 
                          bounds_error=False, fill_value=0)
    
    doppler_audio = interp_func(time_warped)
    
    # Handle NaN values
    doppler_audio = np.nan_to_num(doppler_audio)
    
    return doppler_audio * output_gain


def apply_reverb(audio, fs, room_size=0.7, damping=0.5, wet_level=0.3, output_gain=1.0):
    """
    Apply simple reverb effect using multiple delays.
    
    Args:
        audio: Input audio signal
        fs: Sample rate
        room_size: Room size simulation (0-1)
        damping: High frequency damping (0-1)
        wet_level: Reverb level (0-1)
    """
    # Multiple delay lines for reverb simulation
    delay_times = np.array([0.03, 0.05, 0.07, 0.11, 0.13, 0.17]) * room_size
    decay_factors = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2]) * (1 - damping)
    
    reverb_signal = np.zeros_like(audio)
    
    for delay_time, decay in zip(delay_times, decay_factors):
        delay_samples = int(delay_time * fs)
        if delay_samples > 0 and delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * decay
            reverb_signal += delayed
    
    return (audio + reverb_signal * wet_level) * output_gain


# ==========================================================
# MAIN FUNCTION
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description='Audio Effects Processor')
    parser.add_argument('input_file', nargs='?', help='Input audio file name (looks in data/input/)')
    parser.add_argument('output_file', nargs='?', help='Output audio file base name (saves to data/output/, no extension)')
    parser.add_argument('--wav', action='store_true', help='Generate only WAV output')
    parser.add_argument('--txt', action='store_true', help='Generate only TXT output')
    parser.add_argument('--both', action='store_true', help='Generate both WAV and TXT outputs')
    parser.add_argument('--fs', type=int, default=48000, 
                       help='Sample rate for TXT files (default: 48000)')
    parser.add_argument('--bits', type=int, default=12,
                       help='Bit depth for TXT output (default: 12)')
    
    # Effect parameters
    parser.add_argument('--delay', type=float, nargs=3, metavar=('TIME', 'FEEDBACK', 'MIX'),
                       help='Apply delay: delay_time feedback mix')
    parser.add_argument('--clip', type=float, metavar='THRESHOLD',
                       help='Apply clipping: threshold')
    parser.add_argument('--saturate', type=float, nargs=2, metavar=('DRIVE', 'MIX'),
                       help='Apply saturation: drive mix')
    parser.add_argument('--compress', type=float, nargs=4, metavar=('THRESH', 'RATIO', 'ATTACK', 'RELEASE'),
                       help='Apply compression: threshold ratio attack release')
    parser.add_argument('--distort', type=str, nargs='+', 
                       metavar='PARAM',
                       help='Apply distortion: drive level mix type(tube/fuzz/overdrive/clipping)')
    parser.add_argument('--distort_preset', type=str, metavar='PRESET_NAME',
                       help='Apply distortion preset (use --list_presets to see available)')
    parser.add_argument('--list_presets', action='store_true',
                       help='List available distortion presets')
    parser.add_argument('--noise', type=float, nargs=2, metavar=('LEVEL', 'TYPE'),
                       help='Add noise: level type(0=white,1=pink,2=brown)')
    parser.add_argument('--echo', type=float, nargs=3, metavar=('TIME', 'DECAY', 'NUM'),
                       help='Apply echo: delay_time decay num_echoes')
    parser.add_argument('--doppler', type=float, metavar='MAX_SHIFT',
                       help='Apply Doppler effect: max_shift')
    parser.add_argument('--reverb', type=float, nargs=3, metavar=('SIZE', 'DAMP', 'LEVEL'),
                       help='Apply reverb: room_size damping wet_level')
    
    parser.add_argument('--plot', action='store_true', help='Plot before/after waveforms')
    parser.add_argument('--play', action='store_true', help='Play processed audio')
    
    args = parser.parse_args()
    
    # Handle preset listing
    if args.list_presets:
        presets = get_distortion_presets()
        print("Available distortion presets:")
        print("-" * 40)
        for name, params in presets.items():
            print(f"{name:<18}: drive={params['drive']:<4}, level={params['level']:<4}, "
                  f"mix={params['mix']:<4}, type={params['type']}")
        return
    
    # Check required arguments for file processing
    if not args.input_file or not args.output_file:
        parser.error("input_file and output_file are required for audio processing (use --list_presets to see available presets)")
        
    # Set up input and output directories
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Construct full file paths
    input_path = input_dir / args.input_file
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print(f"Available files in {input_dir}:")
        try:
            files = [f.name for f in input_dir.iterdir() if f.suffix.lower() in ('.wav', '.txt')]
            for f in files:
                print(f"  - {f}")
        except:
            print(f"  Could not list files in {input_dir}")
        return
    
    # Validate output format selection
    if not args.wav and not args.txt and not args.both:
        print("Error: Must specify at least one output format: --wav, --txt, or --both")
        print("Usage examples:")
        print(f"  python audioFX.py {args.input_file} my_processed_audio --wav")
        print(f"  python audioFX.py {args.input_file} my_processed_audio --both")
        return
    
    # Load input audio
    try:
        audio, fs = load_audio(str(input_path), args.fs)
        original_audio = np.copy(audio)
        print(f"Processing audio: {len(audio)} samples at {fs} Hz")
        
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    # Apply effects in order
    if args.delay:
        delay_time, feedback, mix = args.delay
        audio = apply_delay(audio, fs, delay_time, feedback, mix)
        print(f"Applied delay: time={delay_time}s, feedback={feedback}, mix={mix}")

    if args.clip:
        audio = apply_clipping(audio, args.clip)
        print(f"Applied clipping: threshold={args.clip}")

    if args.saturate:
        drive, mix = args.saturate
        audio = apply_saturation(audio, drive, mix)
        print(f"Applied saturation: drive={drive}, mix={mix}")

    if args.compress:
        threshold, ratio, attack, release = args.compress
        audio = apply_compression(audio, threshold, ratio, attack, release, fs)
        print(f"Applied compression: thresh={threshold}, ratio={ratio}")

    if args.distort_preset:
        audio = apply_distortion_preset(audio, args.distort_preset, fs=fs)
        presets = get_distortion_presets()
        if args.distort_preset in presets:
            params = presets[args.distort_preset]
            print(f"Applied preset '{args.distort_preset}': drive={params['drive']}, "
                  f"level={params['level']}, mix={params['mix']}, type={params['type']}")

    if args.distort:
        # Parse distortion arguments
        if len(args.distort) >= 3:
            drive = float(args.distort[0])
            level = float(args.distort[1])
            mix = float(args.distort[2])
            distortion_type = args.distort[3] if len(args.distort) > 3 else 'tube'
        elif len(args.distort) == 2:
            # Backward compatibility: gain, mix
            drive = float(args.distort[0])
            level = 1.0
            mix = float(args.distort[1])
            distortion_type = 'tube'
        else:
            drive = float(args.distort[0])
            level = 1.0
            mix = 1.0
            distortion_type = 'tube'
        
        audio = apply_distortion(audio, drive, level, mix, distortion_type, fs=fs)
        print(f"Applied {distortion_type} distortion: drive={drive}, level={level}, mix={mix}")

    if args.noise:
        level, noise_type_num = args.noise
        noise_types = ['white', 'pink', 'brown']
        noise_type = noise_types[int(noise_type_num) % 3]
        audio = apply_noise(audio, level, noise_type)
        print(f"Added {noise_type} noise: level={level}")

    if args.echo:
        delay_time, decay, num_echoes = args.echo
        audio = apply_echo(audio, fs, delay_time, decay, int(num_echoes))
        print(f"Applied echo: time={delay_time}s, decay={decay}, num={int(num_echoes)}")

    if args.doppler:
        audio = apply_doppler_effect(audio, fs, max_shift=args.doppler)
        print(f"Applied Doppler effect: max_shift={args.doppler}")

    if args.reverb:
        room_size, damping, wet_level = args.reverb
        audio = apply_reverb(audio, fs, room_size, damping, wet_level)
        print(f"Applied reverb: size={room_size}, damp={damping}, wet={wet_level}")
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
        print(f"Normalized by factor: {max_val:.3f}")
    
    # Save processed audio
    try:
        wav_file = output_dir / f"{args.output_file}.wav"
        txt_file = output_dir / f"{args.output_file}.txt"
        
        # Output selection logic
        if args.both:
            save_audio(audio, str(wav_file), fs, args.bits)
            save_audio(audio, str(txt_file), fs, args.bits)
            print(f"Saved both {wav_file} and {txt_file}")
        elif args.wav:
            save_audio(audio, str(wav_file), fs, args.bits)
            print(f"Saved WAV: {wav_file}")
        elif args.txt:
            save_audio(audio, str(txt_file), fs, args.bits)
            print(f"Saved TXT: {txt_file}")
        
    except Exception as e:
        print(f"Error saving audio: {e}")
        return
    
    # Plot comparison
    if args.plot:
        plt.figure(figsize=(12, 8))
        
        t = np.arange(len(original_audio)) / fs
        
        plt.subplot(2, 1, 1)
        plt.plot(t, original_audio)
        plt.title('Original Audio')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        t_processed = np.arange(len(audio)) / fs
        plt.plot(t_processed, audio)
        plt.title('Processed Audio')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Play audio
    if args.play:
        try:
            import sounddevice as sd
            print("Playing processed audio...")
            sd.play(audio.astype(np.float32), fs)
            sd.wait()
        except ImportError:
            print("sounddevice not available for playback")


if __name__ == "__main__":
    main()
