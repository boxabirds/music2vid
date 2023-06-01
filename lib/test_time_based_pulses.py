from typing import Dict
import pytest
import numpy as np
from lib.time_based_pulses import convert_pulses_to_disco_frames, generate_pulses

def test_pulse_times_one():
    pulse_duration = 0.5
    pulse_amplitude = 0.4
    value_at_rest = 1.00
    times_in_sec = np.array([1.0])

    result = generate_pulses(pulse_duration, pulse_amplitude, value_at_rest, times_in_sec)

    expected_output = {
        "00000.00": "1.00",
        "00001.00": "1.40",
        "00001.50": "1.00"
    }

    assert result == expected_output


def test_pulse_times_simple():
    pulse_duration = 0.5
    pulse_amplitude = 0.4
    value_at_rest = 1.00
    times_in_sec = np.array([1.0, 2.0, 3.0, 5.0])

    result = generate_pulses(pulse_duration, pulse_amplitude, value_at_rest, times_in_sec)

    expected_output = {
        "00000.00": "1.00",
        "00001.00": "1.40",
        "00001.50": "1.00",
        "00002.00": "1.40",
        "00002.50": "1.00",
        "00003.00": "1.40",
        "00003.50": "1.00",
        "00005.00": "1.40",
        "00005.50": "1.00"
    }

    assert result == expected_output

def test_pulse_times_overlapping_one():
    pulse_duration = 0.5
    pulse_amplitude = 0.4
    value_at_rest = 1.00
    times_in_sec = np.array([1.0, 2.0, 2.1, 3.0, 5.0])

    result = generate_pulses(pulse_duration, pulse_amplitude, value_at_rest, times_in_sec)

    expected_output = {
        "00000.00": "1.00",
        "00001.00": "1.40",
        "00001.50": "1.00",
        "00002.00": "1.40",
        "00002.05": "1.32",
        "00002.10": "1.40",
        "00002.60": "1.00",
        "00003.00": "1.40",
        "00003.50": "1.00",
        "00005.00": "1.40",
        "00005.50": "1.00"
    }
    assert result == expected_output

def test_pulse_times_overlapping_many():
    pulse_duration = 0.5
    pulse_amplitude = 0.4
    value_at_rest = 1.01
    times_in_sec = np.array([1.0, 2.0, 2.1, 2.3, 3.0, 5.0])

    result = generate_pulses(pulse_duration, pulse_amplitude, value_at_rest, times_in_sec)

    expected_output = {
        "00000.00": "1.01",
        "00001.00": "1.40",
        "00001.50": "1.01",
        "00002.00": "1.40",
        "00002.05": "1.32",
        "00002.10": "1.40",
        "00002.20": "1.24",
        "00002.30": "1.40",
        "00002.80": "1.01",
        "00003.00": "1.40",
        "00003.50": "1.01",
        "00005.00": "1.40",
        "00005.50": "1.01"
    }
    assert result == expected_output

def test_convert_to_frames():
    input:Dict[str,str] = {
        "00000.00": "1.01",
        "00001.00": "1.40",
        "00001.50": "1.01",
        "00002.00": "1.40",
        "00002.05": "1.32",
        "00002.10": "1.40",
        "00002.20": "1.24",
        "00002.30": "1.40",
        "00002.80": "1.01",
        "00003.00": "1.40",
        "00003.50": "1.01",
        "00005.00": "1.40",
        "00005.50": "1.01"
    }

    expected_output = "0: (1.01), 24: (1.40), 36: (1.01), 48: (1.40), 50: (1.32), 51: (1.40), 53: (1.24), 56: (1.40), 68: (1.01), 72: (1.40), 84: (1.01), 120: (1.40), 132: (1.01)"
    actual_output = convert_pulses_to_disco_frames(input, fps=24)
    assert actual_output == expected_output