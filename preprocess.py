import pretty_midi
import os

def isolate_track(midi_obj, instrument_name):
    tracks = []
    midi_tracks = []
    piano_track = None
    for instrument in midi_obj.instruments:
        if instrument_name in instrument.name:
            piano_track = instrument
            tracks.append(piano_track)    

    if not tracks:
        print(f"{instrument_name} track not found in the input midi object.")
    else:
        for track in tracks:
            new_midi = pretty_midi.PrettyMIDI()
            new_midi.instruments.append(track)
            midi_tracks.append(new_midi)
        return midi_tracks
    return None


path=("data/POP909-Dataset/POP909")
track = 'MELODY'
for folder in os.listdir(path):
    if '.DS_Store' not in folder:
        if 'index' not in folder:

            for i in os.listdir(path+'/'+folder):
                full_path = path+'/'+folder+'/'+i
                # print(full_path)
                if ".mid" in i:
                    print(f'-- Working on song {i} --')
                    pm = pretty_midi.PrettyMIDI(full_path)
                    pm_tracks = isolate_track(pm, track)
                    if pm_tracks:

                        for ind, pm in enumerate(pm_tracks):
                            
                            pm.write(f"data/melody/{i}")


#Melody, Vocal, Clarinet, Lead - Flute, voc