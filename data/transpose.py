#converts all midi files in the current folder
import glob
import music21
import mido



def transpose_to_c_major(midi_file):

    majors = dict([("A-", 4),("A", 3),("A#",2),("B-", 2),("B", 1),("C", 0),("C#",-1),("D-", -1),("D", -2),("D#",-3),("E-", -3),("E", -4),("F", -5),("F#",6),("G-", 6),("G", 5),("G#",4)])
    minors = dict([("A-", 1),("A", 0),("A#",-1),("B-", -1),("B", -2),("C", -3),("C#",-4),("D-", -4),("D", -5),("D#",6),("E-", 6),("E", 5),("F", 4),("F#",3),("G-", 3),("G", 2),("G#",1)])

    score = music21.converter.parse(midi_file)
    key = score.analyze('key')

    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
        
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    # Load the MIDI file
    mid = mido.MidiFile(midi_file)

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'note_on' or msg.type == 'note_off':
                msg.note += halfSteps

    mid.save(f'transposed/C_{midi_file}')


#os.chdir("./")
for idx, file in enumerate(glob.glob("*.mid")):
    transpose_to_c_major(file)
    print("Working.. "+str(idx/909*100)+"%", end='\r')
    
