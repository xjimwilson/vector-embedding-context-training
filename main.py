import saveloadfiles, train, shared, usertest

text, choice, loaded, packaged = None, None, None, None

if __name__ == "__main__": # prevents pickles re-running top level script

    def calcsize(bytesize):
        convsize = bytesize, "bytes"
        if bytesize >= 1000: #1kb:
            convsize = f"{int(bytesize / 1000)} KB"
        if bytesize >= 1000000: #1mb:
            convsize = f"{int(bytesize / 1000000)} MB"
        if bytesize >= 1000000000: #1gb:
            convsize = f"{int(bytesize / 1000000000)} GB"
        return convsize

    while choice != 't' and choice != 'g' and choice != 'm':
        choice = input("Enter T to train, M to manually test, or G to generate:\n").lower()
        
    if choice == 't':
        fileinput = str(input("Enter file name:\n"))
        file = saveloadfiles.readfile(fileinput)

        if file != None:
            print(f"Training on {fileinput}...")

            matrix, wordtoid, idtoword = train.embedding(file)

            print("Saving to .npz file...")
            saveloadfiles.savefile(fileinput, matrix, wordtoid, idtoword)

            if fileinput == "":  
                fileinput = "fulltrained"
            fileinput.replace("/","")

            print(f"Successfully trained! Saved knowledge in memory/{fileinput}.npz, with size of {calcsize(saveloadfiles.getfilesize())}")

    elif choice == 'g':
        while loaded == None:
            file = str(input("Enter the file to load:\n"))
            loaded = saveloadfiles.loadfile(file)

        print("File successfully loaded!")

        while True:
            prompt = input("\n")

            generate.choosefirstword(prompt) #starts generation

            print("\n\nWords:", str(shared.wordcounter).replace("\\n","\n"))

    elif choice == 'm':
        while loaded == None:
            file = str(input("Enter the file to load:\n"))
            loaded = saveloadfiles.loadfile(file)
        test = input("\nEnter V to visualise, N for neighbours, A for analogy\n").strip().lower()
        if test == 'n':
            while True:
                word = input("\nEnter word:\n")

                usertest.nearestneighbours(word, shared.matrix, shared.wordtoid, shared.idtoword)
        elif test == 'v':
            while True:
                words = input("\nEnter words:\n")
                
                usertest.visualise2d(shared.matrix, shared.idtoword, words)
        elif test == 'a':
            while True:
                a = input("\nEnter relational word A:\n")
                b = input("\nEnter relational word B:\n")
                c = input("\nEnter test word:\n")

                usertest.analogy(a,b,c, shared.matrix, shared.wordtoid, shared.idtoword)
