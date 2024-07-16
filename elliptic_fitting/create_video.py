import pydub
import pandas as pd
from pydub.generators import Sine
from pydub import AudioSegment
from pathlib import Path
import numpy as np
from pydub.generators import Sine
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips



def generate_sine_wave(duration_ms, frequency_hz):
    """
    Generate a sine wave with the specified duration and frequency.

    Parameters:
    - duration_ms: Duration of the sound in milliseconds.
    - frequency_hz: Frequency of the sine wave in Hz.

    Returns:
    - An AudioSegment object representing the generated sine wave.
    """
    # Generate the sine wave
    sine_wave = Sine(frequency_hz).to_audio_segment(duration=duration_ms)
    return sine_wave


def format_sound_writes(sound_writes_df:pd.DataFrame, normal:[int], ):
    # assign sounds to trials
    sound_writes_df = sound_writes_df.drop_duplicates(subset='Timestamp', keep='first').copy()
    sound_writes_df['Trial_Number'] = np.full_like(sound_writes_df.index, -1)
    sound_writes_diff = sound_writes_df['Time_diff'] = sound_writes_df['Timestamp'].diff()
    # print(sound_writes_diff[:10])
    long_dt = np.squeeze(np.argwhere(sound_writes_diff.values > 1))
    for n, idx in enumerate(long_dt):
        sound_writes_df.loc[idx:, 'Trial_Number'] = n
    sound_writes_df['Trial_Number'] = sound_writes_df['Trial_Number'] + 1
    sound_writes_df['Time_diff'] = sound_writes_df['Timestamp'].diff()
    sound_writes_df['Payload_diff'] = sound_writes_df['Payload'].diff()
    base_pip_idx = normal[0] - (-2 if (normal[0] - normal[1] > 0) else 2)  # base to pip gap
    non_base_pips = sound_writes_df.query('Payload > 8 & Payload != @base_pip_idx')['Payload'].unique()
    sound_writes_df['pattern_pips'] = np.any((sound_writes_df['Payload_diff'] == (-2 if (normal[0] - normal[1] > 0) else 2),
                                              sound_writes_df['Payload'].isin(non_base_pips)), axis=0)
    sound_writes_df['pattern_start'] = sound_writes_df['pattern_pips'] * sound_writes_df['pattern_pips'].diff() > 0
    sound_writes_df['pip_counter'] = np.zeros_like(sound_writes_df['pattern_start']).astype(int)
    for i in sound_writes_df.query('pattern_start == True').index:
        sound_writes_df.loc[i:i + 3, 'pip_counter'] = np.cumsum(sound_writes_df['pattern_pips'].loc[i:i + 3]) \
                                                      * sound_writes_df['pattern_pips'].loc[i:i + 3]

    return sound_writes_df


if __name__ == "__main__":

    vid = Path(r'X:\Dammy\mouse_pupillometry\mouse_hf\DO79_240124_000\DO79_240124_body.mp4')
    vid_ts_path = Path(r'X:\Dammy\harpbins\DO79_HitData_240124a_event_data_92.csv')
    video_timestamps = pd.read_csv(vid_ts_path)
    # video_timestamps['Timestamp'] = video_timestamps['Timestamp'] /1e6

    sound_writes = pd.read_csv(r'X:\Dammy\harpbins\DO79_SoundData_240124a_write_indices.csv')
    sound_writes = format_sound_writes(sound_writes, normal=[10,15,20,25])
    print(sound_writes['Timestamp'].iloc[0], video_timestamps['Timestamp'].values[0])
    print(sound_writes['Timestamp'].iloc[0]-video_timestamps['Timestamp'].values[0])
    sound_writes['Timestamp'] = sound_writes['Timestamp'] - video_timestamps['Timestamp'].iloc[0]


    sounddir = Path(r'C:\bonsai\bonsaiprotocols\Sound\forvid')

    duration_ms = 150  # Duration of each sound in milliseconds
    starting_frequency_hz = 740  # Frequency of Middle C (C4) in Hz

    # Generate sine waves for 22 semitones from Middle C
    sine_waves = [
        generate_sine_wave(duration_ms, starting_frequency_hz * (2 ** (i / 12)))
        for i in range(22)
    ]
    # Example usage:
    # Export the generated sounds to files
    for i, sine_wave in enumerate(sine_waves):
        frequency = starting_frequency_hz * (2 ** (i / 12))
        file_name = f"{i + 8}_{frequency:.2f}Hz.wav"
        sine_wave.export(sounddir/file_name, format="wav") #if not (sounddir/file_name).is_file() else None
    sound_dict = {f"{sound.stem.split('_')[0]}": AudioSegment.from_file(str(sound)) for sound in sounddir.iterdir()}

    #
    # Load the video and audio files
    video_clip = VideoFileClip(str(vid),)
    video_clip = video_clip.without_audio()
    first_pattern_trial = sound_writes.query('Payload == 10')['Trial_Number'].iloc[4]
    # first_pattern_trial = 2
    vid_start_time = sound_writes.query('Trial_Number == @first_pattern_trial')['Timestamp'].iloc[0]
    vid_end_time = sound_writes.query('Trial_Number == @first_pattern_trial+2')['Timestamp'].iloc[-1]
    sound_writes_for_vid = sound_writes.query('Timestamp >= @vid_start_time & Timestamp <= @vid_end_time')
    print(f'vid length: {vid_end_time - vid_start_time}')

    # Set the embedding times (in seconds)
    embedding_times = np.hstack([sound_writes_for_vid['Timestamp'].values,
                                 sound_writes_for_vid['Timestamp'].values[-1]+1])
    embedding_sounds = sound_writes_for_vid['Payload'].values

    # Loop through each embedding time
    clips = []
    stream = AudioSegment.silent(0)
    for i,(embedding_time, embedding_sound) in enumerate(zip(embedding_times, embedding_sounds)):
        # Set the starting time of the audio clip in the video
        start_time = embedding_time
        audio_clip = sound_dict[str(embedding_sound)]
        # Set the end time of the audio clip in the video
        # Set the end time of the audio clip in the video
        # end_time = start_time + min(audio_clip.duration, video_clip.duration - embedding_time)
        # end_time = start_time + min(audio_clip.duration, video_clip.duration - embedding_time)

        # Embed the audio clip into the video
        # clips.append(video_clip.set_audio(audio_clip).subclip(start_time,embedding_times[i+1]))

        if embedding_sound not in [5,3]:
            pulse_dur = len(audio_clip)
        else:
            pulse_dur = len(audio_clip)
        t_to_next = int((embedding_times[i + 1] - start_time)*1000)
        if embedding_sound in [3]:
            stream = stream.append(audio_clip[:t_to_next], crossfade=0)
        else:
            stream = stream.append(audio_clip, crossfade=0)
        stream = stream.append(AudioSegment.silent(t_to_next-pulse_dur),
                               crossfade=0)
        # clips.append(video_clip.set_audio(audio_stream).subclip(start_time, end_time)
        # break
    stream.export(r'D:\output.wav', format="wav")
    stream_clip = AudioFileClip(r'D:\output.wav')
    video_clip = video_clip.subclip(vid_start_time,vid_start_time+stream_clip.duration)

    video_clip = video_clip.set_audio(stream_clip,)
    video_clip.write_videofile("D:\output.mp4", codec="libx264", audio_codec="aac")

    # concatenated_clips = concatenate_videoclips(clips)
    # concatenated_clips.write_videofile("D:\output.mp4", codec="libx264", audio_codec="aac")
