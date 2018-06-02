import librosa
import pretty_midi
import numpy as np
from scipy.misc import imresize
def load(msd_id, matches):
    try:
        # Compute constant-Q spectrogram.
        audio, fs = librosa.ms(msd_id+".wav", sr=None)
        cqt = librosa.logamplitude(librosa.cqt(audio, real=False))
        cqt = librosa.util.normalize(cqt)

        # TODO Synthesize MIDI. We could use this for the Lakh MIDI files without 7digital preview MP3s.
        # midi_audio = pm.fluidsynth(fs)
        # display(Audio(midi_audio, rate=fs))

        # Average existing MIDI annotations for the audio into a single matrix.
        piano_roll = None
        midi_md5 = max(matches, key=lambda x: x[1])
        """
        for midi_md5, score in matches.items():
            pm = pretty_midi.PrettyMIDI(midi_path(msd_id, midi_md5, 'aligned'))
            x = pm.get_piano_roll()
            if x.shape[1] > 0:
                if piano_roll is not None:
                    x = imresize(x, size=piano_roll.shape)
                    piano_roll = np.dstack((piano_roll, x))
                else:
                    piano_roll = pm.get_piano_roll()[..., np.newaxis]
        """
        pm = pretty_midi.PrettyMIDI(msd_id+".mid")
        piano_roll = pm.get_piano_roll()[..., np.newaxis]
        piano_roll = np.average(piano_roll, axis=2)

        # Use 7 octaves starting from C1.
        piano_roll = piano_roll[12:96]
        piano_roll = imresize(piano_roll, size=cqt.shape)
        piano_roll = piano_roll.astype(
            np.float) / 255  # Normalize to [0.0, 1.0].
        return cqt, piano_roll
    except Exception as e:
        print(e)
        raise e