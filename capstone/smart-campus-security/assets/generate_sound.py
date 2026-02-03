"""
Generate a simple alert sound for the system
Creates a beep sound using numpy and scipy
"""

import numpy as np
from scipy.io import wavfile
from pathlib import Path

def generate_alert_sound(filename: str = "assets/alert.wav", duration: float = 0.5, frequency: int = 1000):
    """
    Generate a simple beep sound
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        frequency: Frequency in Hz
    """
    # Create directory
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Sample rate
    sample_rate = 44100
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate beep (sine wave with envelope)
    beep = np.sin(2 * np.pi * frequency * t)
    
    # Apply envelope (fade in/out)
    envelope = np.ones_like(t)
    fade_samples = int(0.05 * sample_rate)  # 50ms fade
    
    # Fade in
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    
    # Fade out
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    # Apply envelope
    beep = beep * envelope
    
    # Normalize to 16-bit range
    beep = np.int16(beep * 32767 * 0.5)  # 50% volume
    
    # Save as WAV
    wavfile.write(filename, sample_rate, beep)
    
    print(f"Alert sound generated: {filename}")

if __name__ == "__main__":
    generate_alert_sound()
