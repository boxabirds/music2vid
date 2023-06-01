from typing import Dict
import numpy as np
from collections import OrderedDict

# this gives us 27h and 30 minutes of animation. Good enough.
TIMESTAMP_FORMAT = "{:0>8.2f}" # 5 characters, 2 decimal places e.g 00001.00
PULSE_FORMAT = "{:.2f}" # 2 decimal places e.g. 1.00
def format_timestamp(time_in_sec: float) -> str:
    return TIMESTAMP_FORMAT.format(time_in_sec)

def generate_pulses(pulse_duration, pulse_amplitude, value_at_rest, times_in_sec) -> Dict[str, str]:
    # Initialize the result dictionary with the value at rest

    result = {format_timestamp(0.0): PULSE_FORMAT.format(value_at_rest)}

    # Iterate over the times in seconds
    for idx, time_in_sec in enumerate(times_in_sec):
        pulse_start_key = format_timestamp(time_in_sec)
        result[pulse_start_key] = PULSE_FORMAT.format(1.0 + pulse_amplitude)

        # if we're not at the last time in the list, check the next item. if it's within the pulse duration, add a keyframe for the end of the pulse
        if idx + 1 < len(times_in_sec):
            next_time_in_sec = times_in_sec[idx + 1]

            # the next time is within the pulse duration so we will add a keyframe that represents the decay of the pulse half way between the start of the pulse and the next pulse
            pulse_gap = next_time_in_sec - time_in_sec
            if pulse_gap < pulse_duration:
                # insert a partial pulse decay keyframe between the two pulses
                pulse_end_key = format_timestamp(time_in_sec + pulse_gap / 2)
                result[pulse_end_key] = PULSE_FORMAT.format(1.0 + pulse_amplitude - pulse_amplitude * (pulse_gap / pulse_duration) )
                print(f"Adding partial pulse decay keyframe at {pulse_end_key} with value {result[pulse_end_key]}")
            else:
                pulse_end_key = format_timestamp(time_in_sec + pulse_duration)
                result[pulse_end_key] = PULSE_FORMAT.format(value_at_rest)
        else:
            pulse_end_key = format_timestamp(time_in_sec + pulse_duration)
            result[pulse_end_key] = PULSE_FORMAT.format(value_at_rest)

    return result

def convert_pulses_to_disco_frames(pulses: Dict[str, str], fps: int) -> str:
    # Convert the pulse times to key frame numbers
    kf = OrderedDict()
    for pulse_time, pulse_value in pulses.items():

        # convert time into a key frame number e.g. fps = 24, "00002.05": "1.32", => frame 49
        key_frame_number = str(int(np.ceil(float(pulse_time) * fps)))
        kf[key_frame_number] = pulse_value

    # Generate the final key frame string
    kf_disco_string = ", ".join(f"{key}: ({value})" for key, value in kf.items())

    return kf_disco_string