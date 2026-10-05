import numpy as np
import matplotlib.pyplot as plt

def nearestneighbours(word, matrix, wordtoid, idtoword, topn=10):
    
    if word not in wordtoid:
        print(f"Cannot find word '{word}' in dataset")
        return

    wordid = wordtoid[word]
    vector = matrix[wordid]

    # using cosine similarity against all dimensions in matrix
    normals = np.linalg.norm(matrix, axis=1)
    similarities = (matrix @ vector) / (normals * np.linalg.norm(vector) + 1e-9)

    #get indices sorted by similarity descending while skipping original word
    ranked = np.argsort(-similarities)
    ranked = [i for i in ranked if i != wordid][:topn]

    print()

    for wordid in ranked:
        print(f"{idtoword[wordid]:15s} sim={similarities[wordid]:.8f}")

def visualise2d(matrix, idtoword, words=None, topn=200):
    if words is not None:
        if isinstance(words, str):
            words = words.split()  # avoid silently iterating character-by-character
 
        wordtoid_lookup = {v: k for k, v in idtoword.items()}
        ids = [wordtoid_lookup[w] for w in words if w in wordtoid_lookup]
    else:
        ids = list(range(min(topn, matrix.shape[0])))
 
    subset = matrix[ids]
    centred = subset - subset.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ vt[:2].T
 
    plt.figure(figsize=(10, 8))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.6)
    for i, wordid in enumerate(ids):
        plt.annotate(idtoword[wordid], (projected[i, 0], projected[i, 1]), fontsize=8)
    plt.title("Word embeddings (PCA projection)")
    plt.show()

def analogy(a, b, c, matrix, wordtoid, idtoword, topn=10):
    """
    Classic word2vec analogy test: a is to b as c is to ?
    e.g. analogy("man", "king", "woman", ...) should surface "queen"
    """
    for w in (a, b, c):
        if w not in wordtoid:
            print(f"Cannot find word '{w}' in dataset")
            return
 
    vector = matrix[wordtoid[b]] - matrix[wordtoid[a]] + matrix[wordtoid[c]]
 
    normals = np.linalg.norm(matrix, axis=1)
    similarities = (matrix @ vector) / (normals * np.linalg.norm(vector) + 1e-9)
 
    excluded = {wordtoid[a], wordtoid[b], wordtoid[c]}
    ranked = np.argsort(-similarities)
    ranked = [i for i in ranked if i not in excluded][:topn]
 
    print(f"\n{a} : {b} :: {c} : ?")
    for wordid in ranked:
        print(f"{idtoword[wordid]:15s} sim={similarities[wordid]:.4f}\n")