# Creación del Dataset de Audio

Esta carpeta contiene los scripts necesarios para generar los datos de entrenamiento para el modelo TCN de emulación de efectos de audio.

El proceso se divide en dos pasos principales:

1.  **Generación de Audio Base**: Se crea una señal de audio limpia que servirá como entrada (`input`).
2.  **Aplicación de Efectos**: A la señal limpia se le aplica un efecto (como distorsión, saturación, etc.) para crear la señal objetivo (`output`).

## Scripts

### `audioGenerator.py`

*   **Propósito**: Generar una señal de audio base. Puede ser un tono puro, un barrido de frecuencias (chirp), o cualquier otra señal que sea útil para que la red aprenda la respuesta en frecuencia del efecto.
*   **Uso**:
    ```bash
    # Generar solo archivo WAV
    python audioGenerator.py --outfile rawAudio --wav
    
    # Generar solo archivo TXT
    python audioGenerator.py --outfile rawAudio --txt
    
    # Generar ambos formatos
    python audioGenerator.py --outfile rawAudio --wav --txt
    ```
    
*   **Argumentos importantes**:
    - `--outfile`: Nombre base del archivo (sin extensión). Los archivos se guardan en `../input/`
    - `--wav`: Genera archivo WAV (formato de audio estándar)
    - `--txt`: Genera archivo TXT (datos cuantizados para procesamiento)
    - Al menos uno de `--wav` o `--txt` debe especificarse

### `audioFX.py`

*   **Propósito**: Aplicar uno o varios efectos de audio a una señal de entrada. Este script lee un fichero desde `data/input/`, le aplica un procesamiento (ej: distorsión de amplificador) y guarda el resultado en `data/output/`.
*   **Uso**:
    ```bash
    # Aplicar preset de distorsión y generar WAV
    python audioFX.py rawAudio.wav fxAudio --wav --distort_preset light_overdrive
    
    # Generar ambos formatos con efectos múltiples
    python audioFX.py rawAudio.wav fxAudio --both --distort_preset vintage_fuzz --reverb 0.7 0.5 0.3
    
    # Solo archivo TXT con compresión
    python audioFX.py rawAudio.wav fxAudio --txt --compress 0.5 4.0 0.01 0.1
    ```

*   **Argumentos importantes**:
    - `input_file`: Nombre del archivo en `data/input/` (ej: `rawAudio.wav`)
    - `output_file`: Nombre base para guardar en `data/output/` (sin extensión)
    - `--wav`: Genera archivo WAV procesado
    - `--txt`: Genera archivo TXT procesado  
    - `--both`: Genera ambos formatos
    - Al menos uno de `--wav`, `--txt` o `--both` debe especificarse

*   **Efectos disponibles**: `--distort_preset`, `--delay`, `--reverb`, `--compress`, `--clip`, `--saturate`, etc.
*   **Ver presets**: `python audioFX.py --list_presets`

