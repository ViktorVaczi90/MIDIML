from midiutil import MIDIFile

degrees = range(40, 73)  # MIDI note number
track = 0
channel = 0
time = 0  # In beats
duration = 4  # In beats
tempo = 120  # In BPM
# volume   = 19  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
# automatically)
MyMIDI.addTempo(track, time, tempo)

for j, volume in enumerate(range(20, 21)):
    for i, pitch in enumerate(degrees):
        MyMIDI.addNote(track, channel, pitch, time + 8 * i + j * len(degrees) * 8, duration, volume)

with open("MIDIproba.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
