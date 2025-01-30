import librosa
import numpy as np
from pydub import AudioSegment

from clearervoice48k.dataloader.misc import (get_file_extension,
                                             read_and_config_file)

EPS = 1e-6
MAX_WAV_VALUE_16B = 32768.0
MAX_WAV_VALUE_32B = 2147483648.0


def read_audio(file_path):
    """
    Use AudioSegment to load audio from all supported audio input format
    """

    try:
        audio = AudioSegment.from_file(file_path)
        return audio
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def audioread(path, sampling_rate, use_norm):
    """
    Reads an audio file from the specified path, normalizes the audio,
    resamples it to the desired sampling rate (if necessary), and ensures it is single-channel.

    Parameters:
    path (str): The file path of the audio file to be read.
    sampling_rate (int): The target sampling rate for the audio.
    use_norm (bool): The flag for specifying whether using input audio normalization

    Returns:
    numpy.ndarray: The processed audio data, normalized, resampled (if necessary),
                   and converted to mono (if the input audio has multiple channels).
    """

    # Read audio data and its sample rate from the file.
    audio_info = {}
    ext = get_file_extension(path).replace('.', '')
    audio_info['ext'] = ext

    try:
        data = AudioSegment.from_file(path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    data = read_audio(path)

    audio_info['sample_rate'] = data.frame_rate
    audio_info['channels'] = data.channels
    audio_info['sample_width'] = data.sample_width

    data_array = np.array(data.get_array_of_samples())
    if max(data_array) > MAX_WAV_VALUE_16B:
        audio_np = data_array / MAX_WAV_VALUE_32B
    else:
        audio_np = data_array / MAX_WAV_VALUE_16B

    audios = []
    # Check if the audio is stereo
    if audio_info['channels'] == 2:
        audios.append(audio_np[::2])  # Even indices (left channel)
        audios.append(audio_np[1::2])  # Odd indices (right channel)
    else:
        audios.append(audio_np)

    # Normalize the audio data.
    audios_normed = []
    scalars = []
    for audio in audios:
        if use_norm:
            audio_normed, scalar = audio_norm(audio)
            audios_normed.append(audio_normed)
            scalars.append(scalar)
        else:
            audios_normed.append(audio)
            scalars.append(1)
    # Resample the audio if the sample rate is different from the target sampling rate.
    if audio_info['sample_rate'] != sampling_rate:
        index = 0
        for audio_normed in audios_normed:
            audios_normed[index] = librosa.resample(audio_normed, orig_sr=audio_info['sample_rate'], target_sr=sampling_rate)
            index = index + 1

    # Return the processed audio data.
    return audios_normed, scalars, audio_info


def audio_norm(x):
    """
    Normalizes the input audio signal to a target Root Mean Square (RMS) level, 
    applying two stages of scaling. This ensures the audio signal is neither too quiet 
    nor too loud, keeping its amplitude consistent.

    Parameters:
    x (numpy.ndarray): Input audio signal to be normalized.

    Returns:
    numpy.ndarray: Normalized audio signal.
    """

    # Compute the root mean square (RMS) of the input audio signal.
    rms = (x ** 2).mean() ** 0.5

    # Calculate the scalar to adjust the signal to the target level (-25 dB).
    scalar = 10 ** (-25 / 20) / (rms + EPS)

    # Scale the input audio by the computed scalar.
    x = x * scalar

    # Compute the power of the scaled audio signal.
    pow_x = x ** 2

    # Calculate the average power of the audio signal.
    avg_pow_x = pow_x.mean()

    # Compute RMS only for audio segments with higher-than-average power.
    rmsx = pow_x[pow_x > avg_pow_x].mean() ** 0.5

    # Calculate another scalar to further normalize based on higher-power segments.
    scalarx = 10 ** (-25 / 20) / (rmsx + EPS)

    # Apply the second scalar to the audio.
    x = x * scalarx

    # Return the doubly normalized audio signal.
    return x, 1 / (scalar * scalarx + EPS)


class DataReader(object):
    """
    A class for reading audio data from a list of files, normalizing it, 
    and extracting features for further processing. It supports extracting 
    features from each file, reshaping the data, and returning metadata 
    like utterance ID and data length.

    Parameters:
    args: Arguments containing the input path and target sampling rate.

    Attributes:
    file_list (list): A list of audio file paths to process.
    sampling_rate (int): The target sampling rate for audio files.
    """

    def __init__(self, args):
        # Read and configure the file list from the input path provided in the arguments.
        # The file list is decoded, if necessary.
        self.file_list = read_and_config_file(args, args.input_path, decode=True)

        # Store the target sampling rate.
        self.sampling_rate = args.sampling_rate

        # Store the args file
        self.args = args

    def __len__(self):
        """
        Returns the number of audio files in the file list.

        Returns:
        int: Number of files to process.
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Retrieves the features of the audio file at the given index.

        Parameters:
        index (int): Index of the file in the file list.

        Returns:
        tuple: Features (inputs, utterance ID, data length) for the selected audio file.
        """
        if self.args.task == 'target_speaker_extraction':
            if self.args.network_reference.cue == 'lip':
                return self.file_list[index]
        return self.extract_feature(self.file_list[index])

    def extract_feature(self, path):
        """
        Extracts features from the given audio file path.

        Parameters:
        path (str): The file path of the audio file.

        Returns:
        inputs (numpy.ndarray): Reshaped audio data for further processing.
        utt_id (str): The unique identifier of the audio file, usually the filename.
        length (int): The length of the original audio data.
        """
        # Extract the utterance ID from the file path (usually the filename).
        utt_id = path.split('/')[-1]
        use_norm = False

        # We suggest to use norm for 'FRCRN_SE_16K' and 'MossFormer2_SS_16K' models
        if self.args.network in ['FRCRN_SE_16K', 'MossFormer2_SS_16K']:
            use_norm = True

        # Read and normalize the audio data, converting it to float32 for processing.
        audios_norm, scalars, audio_info = audioread(path, self.sampling_rate, use_norm)

        if self.args.network in ['MossFormer2_SR_48K']:
            audio_info['sample_rate'] = self.sampling_rate

        for i in range(len(audios_norm)):
            audios_norm[i] = audios_norm[i].astype(np.float32)
            # Reshape the data to ensure it's in the format [1, data_length].
            audios_norm[i] = np.reshape(audios_norm[i], [1, audios_norm[i].shape[0]])

        # Return the reshaped audio data, utterance ID, and the length of the original data.
        return audios_norm, utt_id, audios_norm[0].shape[1], scalars, audio_info
