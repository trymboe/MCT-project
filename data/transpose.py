#converts all midi files in the current folder
import glob
import os
import music21



def transpose_to_c_major(midi_file):
    score = music21.converter.parse(midi_file)
    print(f"Working on {midi_file}")
    key = score.analyze("key")
    steps_to_c = music21.interval.Interval(key.tonic, music21.pitch.Pitch("C"))

    transposed_score = score.transpose(steps_to_c)

    transposed_score.write("midi", f"transposed/C_{midi_file}")


#os.chdir("./")
for file in glob.glob("*.mid"):
    transpose_to_c_major(file)
