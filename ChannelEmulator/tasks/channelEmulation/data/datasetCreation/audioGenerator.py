# generar_audio.py
# Generador de señales de entrenamiento (traducción de MATLAB a Python)

import argparse
import numpy as np
from pathlib import Path
from scipy.signal import chirp, firwin, lfilter, square, sawtooth
import soundfile as sf
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURACIÓN POR DEFECTO
# ==========================================================
BITS_DEFAULT        = 12
FS_KHZ_DEFAULT      = 48
NOMBRE_ARCHIVO_DEF  = 'rawAudio'

# Sweep senoidal
F0_SWP   = 20.0
F1_SWP   = 20000.0
DUR_SWP  = 2.0

# Silencios
DUR_SIL = 0.2

# Ruido blanco fijo
F_LOW   = 20.0
F_HIGH  = 20000.0
DUR_RND = 3.0

# Ruido blanco modulado (cuadrada)
F_CUAD     = 6.0
N_PER_CUAD = 18    # 3 s

# Ruido blanco modulado (sierra)
PERIOD_SAW  = 0.5
N_PER_SAW   = 6
AMP_MAX_SAW = 1.0
AMP_MIN_SAW = 0.3

# Sweep triangular cromático
F_TRI1_MIN = 27.5
F_TRI1_MAX = 4186.0
DUR_TRI1   = 5.0

F_TRI2_MIN = 4186.0
F_TRI2_MAX = 27.5
DUR_TRI2   = 5.0


# ==========================================================
# UTILIDADES
# ==========================================================
def n_samp(fs, dur):
    return int(round(fs * dur))

def linspace_dur(fs, dur):
    N = n_samp(fs, dur)
    return np.linspace(0.0, dur, N, endpoint=False)

def zeros_dur(fs, dur):
    return np.zeros(n_samp(fs, dur), dtype=np.float64)

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


# ==========================================================
# FUNCIONES DE SEÑAL
# ==========================================================
def sweep_senoidal(t, f0, f1, dur):
    return 0.7 * chirp(t, f0=f0, f1=f1, t1=dur, method='linear')

def ruido_blanco_bandpass(t, fs, f_low, f_high, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    wgn = rng.standard_normal(len(t))
    taps = firwin(1025, [f_low, f_high], pass_zero=False, fs=fs)
    ruido = lfilter(taps, [1.0], wgn)
    m = np.max(np.abs(ruido))
    if m > 0:
        ruido = ruido / m
    return ruido

def ruido_blanco_modulado(t, fs, f_low, f_high, moduladora, rng=None):
    base = ruido_blanco_bandpass(t, fs, f_low, f_high, rng=rng)
    ruido = base * moduladora.reshape(-1)
    m = np.max(np.abs(ruido))
    if m > 0:
        ruido = ruido / m
    return ruido

def triangular_semitone_sweep(f_start, f_end, dur, fs, amp=1.0):
    up = f_end > f_start
    step = 2 ** (np.sign(f_end - f_start) / 12.0)
    freqs = [float(f_start)]
    if up:
        while freqs[-1] * step <= f_end + 1e-9:
            freqs.append(freqs[-1] * step)
    else:
        while freqs[-1] * step >= f_end - 1e-9:
            freqs.append(freqs[-1] * step)
    N = len(freqs)

    slot = dur / N
    residual = 0.0
    y_list = []
    for f in freqs:
        t_nom = slot + residual
        n_periods = int(round(f * t_nom))
        if n_periods < 1:
            n_periods = 1
        n_samples = int(round(n_periods * fs / f))
        t_local = np.arange(n_samples) / fs
        seg = amp * sawtooth(2.0 * np.pi * f * t_local, width=0.5)
        y_list.append(seg)
        actual = n_samples / fs
        residual = t_nom - actual

    y = np.concatenate(y_list) if y_list else np.zeros(0)
    L_target = n_samp(fs, dur)
    if len(y) > L_target:
        y = y[:L_target]
    elif len(y) < L_target:
        y = np.pad(y, (0, L_target - len(y)), mode='constant')
    return y

def generate_random_notes(fs, duration, f_min, f_max, rng=None):
    """
    Generate a signal with random musical notes and background noise.
    
    Args:
        fs: Sampling frequency
        duration: Total duration in seconds
        f_min: Minimum frequency for notes
        f_max: Maximum frequency for notes
        rng: Random number generator
    
    Returns:
        Generated signal as numpy array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize parameters inside the function
    num_notes = rng.integers(20, 100)  # Number of notes
    attack_time = 0.05  # Attack time in seconds
    sustain_level = 0.8  # Sustain level
    release_time = 0.1  # Release time in seconds
    note_min_duration = 0.2  # Minimum note duration
    note_max_duration = 1.5  # Maximum note duration
    noise_amp = 0.05  # Background noise amplitude
    lfo_freq = 0.2  # LFO frequency for noise modulation
    
    total_samples = int(fs * duration)
    signal = np.zeros(total_samples)
    
    # Generate background noise modulated by LFO
    t = np.arange(total_samples) / fs
    lfo = 0.5 * (np.sin(2 * np.pi * lfo_freq * t) + 1)  # 0 to 1
    noise = rng.standard_normal(total_samples) * noise_amp * lfo
    signal += noise
    
    # Generate random notes
    for _ in range(num_notes):
        # Random start time
        start_time = rng.uniform(0, duration - note_min_duration)
        
        # Random frequency (musical notes approximation)
        # Use semitones for more musical feel
        semitone_steps = int(np.log2(f_max / f_min) * 12)
        semitone = rng.integers(0, semitone_steps + 1)
        note_freq = f_min * (2 ** (semitone / 12.0))
        
        # Random shape
        shapes = ['sine', 'sawtooth', 'triangle', 'square']
        shape = rng.choice(shapes)
        
        # Random duration
        note_duration = rng.uniform(note_min_duration, min(note_max_duration, duration - start_time))
        
        # Calculate envelope times
        attack_samples = int(attack_time * fs)
        release_samples = int(release_time * fs)
        sustain_samples = int((note_duration - attack_time - release_time) * fs)
        
        if sustain_samples < 0:
            sustain_samples = 0
            attack_samples = int(note_duration * 0.3 * fs)
            release_samples = int(note_duration * 0.3 * fs)
            sustain_samples = int(note_duration * 0.4 * fs)
        
        total_note_samples = attack_samples + sustain_samples + release_samples
        
        if total_note_samples == 0:
            continue
        
        # Generate time array for the note
        t_note = np.arange(total_note_samples) / fs
        
        # Generate waveform
        if shape == 'sine':
            wave = np.sin(2 * np.pi * note_freq * t_note)
        elif shape == 'sawtooth':
            wave = sawtooth(2 * np.pi * note_freq * t_note)
        elif shape == 'triangle':
            wave = sawtooth(2 * np.pi * note_freq * t_note, width=0.5)
        elif shape == 'square':
            wave = square(2 * np.pi * note_freq * t_note)
        
        # Apply ASR envelope
        env = np.zeros(total_note_samples)
        
        # Attack
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Sustain
        if sustain_samples > 0:
            env[attack_samples:attack_samples + sustain_samples] = sustain_level
        
        # Release
        if release_samples > 0:
            env[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
        
        wave *= env
        
        # Add to main signal
        start_sample = int(start_time * fs)
        end_sample = start_sample + total_note_samples
        
        if end_sample > total_samples:
            end_sample = total_samples
            wave = wave[:end_sample - start_sample]
        
        signal[start_sample:end_sample] += wave * 0.7  # Reduce note amplitude
    
    # Normalize the signal
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal /= max_amp
    
    return signal

# ==========================================================
# MAIN
# ==========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=BITS_DEFAULT)
    ap.add_argument("--fs_khz", type=float, default=FS_KHZ_DEFAULT)
    ap.add_argument("--outfile", type=str, default=NOMBRE_ARCHIVO_DEF, help="Nombre base del archivo (sin extensión)")
    ap.add_argument("--wav", action="store_true", help="Exportar en formato WAV")
    ap.add_argument("--txt", action="store_true", help="Exportar en formato TXT")
    ap.add_argument("--plot", action="store_true", help="Mostrar gráfico de la señal")
    ap.add_argument("--seed", type=int, default=None, help="Semilla RNG para reproducibilidad")
    ap.add_argument("--play", action="store_true", help="Reproducir con sounddevice")
    args = ap.parse_args()

    # Verificar que se haya especificado al menos un formato de salida
    if not args.wav and not args.txt:
        print("Error: Debe especificar al menos uno de los argumentos --wav o --txt")
        print("Uso: python audioGenerator.py --outfile mi_audio --wav --txt")
        return

    bits = args.bits
    fs = int(round(args.fs_khz * 1000))
    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()

    # Crear el directorio de salida si no existe
    base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "input"
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de salida: {output_dir}")

    # Crear rutas de archivos completas
    base_filename = args.outfile
    txt_path = output_dir / f"{base_filename}.txt"
    wav_path = output_dir / f"{base_filename}.wav"

    # Construcción de la señal
    silencio0 = zeros_dur(fs, DUR_SIL)

    tSwp = linspace_dur(fs, DUR_SWP)
    sweep_sig = sweep_senoidal(tSwp, F0_SWP, F1_SWP, DUR_SWP)
    silencio1 = zeros_dur(fs, DUR_SIL)
    tRnd = linspace_dur(fs, DUR_RND)
    ruido = ruido_blanco_bandpass(tRnd, fs, F_LOW, F_HIGH, rng=rng)

    silencio2 = zeros_dur(fs, DUR_SIL)

    dur_cuad = N_PER_CUAD / F_CUAD
    tModCuad = linspace_dur(fs, dur_cuad)
    onda_cuad = 0.5 * (square(2*np.pi*F_CUAD*tModCuad) + 1.0)
    ruido_mod_cuad = ruido_blanco_modulado(tModCuad, fs, F_LOW, F_HIGH, onda_cuad, rng=rng)

    silencio3 = zeros_dur(fs, DUR_SIL)

    dur_saw = N_PER_SAW * PERIOD_SAW
    tModSaw = linspace_dur(fs, dur_saw)
    ramp = (tModSaw % PERIOD_SAW) / PERIOD_SAW
    onda_saw = AMP_MAX_SAW - (AMP_MAX_SAW - AMP_MIN_SAW) * ramp
    ruido_mod_saw = ruido_blanco_modulado(tModSaw, fs, F_LOW, F_HIGH, onda_saw, rng=rng)

    silencio4 = zeros_dur(fs, DUR_SIL)

    sweep_tri1 = triangular_semitone_sweep(F_TRI1_MIN, F_TRI1_MAX, DUR_TRI1, fs, amp=0.7)
    sweep_tri2 = triangular_semitone_sweep(F_TRI2_MIN, F_TRI2_MAX, DUR_TRI2, fs, amp=0.7)
    
    silencio5 = zeros_dur(fs, DUR_SIL)

    total_duration = 20.0  # Total duration in seconds
    f_min = 20.0
    f_max = 20000.0
    mots_aleatorias = generate_random_notes(fs, total_duration, f_min, f_max, rng)

    senal = np.concatenate([
        #silencio0, 
        #sweep_sig, 
        #silencio1, 
        ruido, 
        #silencio2,
        #ruido_mod_cuad, 
        #silencio3, 
        #ruido_mod_saw, 
        #silencio4,
        #sweep_tri1, 
        #sweep_tri2, 
        #silencio5,
        #mots_aleatorias
    ])

    # Normalización y cuantización
    senal = clamp(senal, -1.0, 1.0)
    senal_u = np.rint((senal + 1.0) / 2.0 * (2**bits - 1)).astype(np.int64)
    senal_u = clamp(senal_u, 0, 2**bits - 1).astype(np.int64)

    from datetime import datetime
    fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Guardar archivo TXT si se solicitó
    if args.txt:
        header = [
            f"# fs={fs}",
            f"# bits={bits}",
            f"# samples={len(senal_u)}",
            f"# date={fecha}"
        ]
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in header:
                f.write(line + '\n')
            np.savetxt(f, senal_u, fmt='%d', delimiter='\t')
        print(f"Archivo TXT generado: {txt_path} (muestras: {len(senal_u)}, fs={fs} Hz, bits={bits}, fecha={fecha})")

    # Guardar archivo WAV si se solicitó
    if args.wav:
        sf.write(str(wav_path), senal.astype(np.float32), fs, subtype='PCM_16')
        print(f"Archivo WAV generado: {wav_path}")

    if args.plot:
        tt = np.arange(len(senal)) / fs
        plt.figure()
        plt.plot(tt, senal)
        plt.grid(True)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (–1..1)')
        plt.title('Señal generada')
        plt.show()

    if args.play:
        import sounddevice as sd
        sd.play(senal.astype(np.float32), fs)
        sd.wait()


if __name__ == "__main__":
    main()
