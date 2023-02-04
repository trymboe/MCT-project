import os

path = 'data/drums'
for i in os.listdir(path):
    if ".mid" in i:
        if "rock" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"rock"+'/'+i)
        if "pop" in i:
           print(i)
           os.replace(path+'/'+i, path+'/'+"pop"+'/'+i)
        if "hiphop" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"hiphop"+'/'+i)
        if "folk" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"folk"+'/'+i)
        if "jazz" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"jazz"+'/'+i)
        if "funk" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"funk"+'/'+i)
        if "latin" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"latin"+'/'+i)
        if "afrocuban" in i:
           print(i)
           os.replace(path+'/'+i, path+'/'+"afrocuban"+'/'+i)
        if "reggae" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"reggae"+'/'+i)
        if "country" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"country"+'/'+i)
        if "punk" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"punk"+'/'+i)
        if "blues" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"blues"+'/'+i)
        if "soul" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"soul"+'/'+i)
        if "afrobeat" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"afrobeat"+'/'+i)
        if "dance" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"dance"+'/'+i)
        if "neworleans" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"neworleans"+'/'+i)
        if "gospel" in i:
            print(i)
            os.replace(path+'/'+i, path+'/'+"gospel"+'/'+i)
        