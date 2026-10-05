import os
import numpy as np
import shared


def readfile(file):

    if ".txt" not in file:
        file += ".txt"
        
    try:
        f = open(f"datasets/{file}", 'r', encoding="utf-8")
        return f.read()
    except:
        print("Could not find file",file)
        return None

def savefile(file, matrix, wordtoid, idtoword):
    if file == "": file = "fulltrained" #defaults to fulltrained.npz when empty

    else:
        file = file.replace(".txt",'')

        shared.filepath = f"memory/{file}"

        # save as .npz (finding out compression was like discovering fire)
        np.savez_compressed(str(shared.filepath),
            matrix = matrix,
            wordtoid = wordtoid,
            idtoword = idtoword)


def loadfile(file):
    global result

    file = f"memory/{file}"

    if ".npz" not in file:
        file = file + ".npz" # just in case user does not input .npz

    if file == "":
        file = "fulltrained"

    try:
        data = np.load(file, allow_pickle=True)

        shared.matrix = data['matrix']              # normal array, no .item() needed
        shared.wordtoid = data['wordtoid'].item()   # unwrap back into a real dict
        shared.idtoword = data['idtoword'].item()

        data.close()

        print("Sucessfully loaded", file)
        return True #signals it went alright

    except Exception as e:
        print(f"Could not load {file}, {e}")
        return #returns None

def getfilesize():
    return os.path.getsize(f"{shared.filepath}.npz")