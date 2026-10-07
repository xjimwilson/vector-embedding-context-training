import re, cupyx
import numpy as np
import cupy as cp # AWFUL abbreviation
import shared

#rewrote basically everything AGAIN after moving to vector embedding

negativesamples = 5  # how many incorrect words to contrast against per pair
dimensions = 600
matrixscalar = 1 / cp.sqrt(dimensions) #scale down matrixes since they usually blow up
learningrate = 0.0012
batchsize = 50000 #how many pairs per op
subsamplethreshold = 1e-3  # lower for more aggressive dropping of frequent words
negativetablesize = int(1e7)

epochthresh = 0.05 #stops epochs when change in loss falls below value

def tokenize(text): #cleanses text and returns as a massive tuple

    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = text.split()

    wordtoid = {}
    idtoword = {}
    uniquewords = sorted(set(tokens))

    for i, token in enumerate(uniquewords):
        wordtoid[token] = i
        idtoword[i] = token

    tokenids = np.array([wordtoid[t] for t in tokens])

    counts = np.bincount(tokenids, minlength=len(uniquewords))

    return wordtoid, idtoword, tokenids, counts

def subsample(tokenids, counts, threshold=subsamplethreshold):
    # word2vec has frequent word subsampling. without this positive pairs get dominated by words
    # like the, of, and, etc. because they're really frequent, so important words just get ignored and look equal to everything else
 
    freq = counts / counts.sum()
    keepprob = (np.sqrt(freq / threshold) + 1) * (threshold / freq)
    keepprob = np.clip(keepprob, 0, 1)
 
    tokenids = np.array(tokenids)
    randvals = np.random.rand(len(tokenids))
    mask = randvals < keepprob[tokenids]
 
    kept = tokenids[mask]
    print(f"Subsampling kept {int(mask.sum())}/{len(tokenids)} tokens "
          f"({float(mask.sum()) / len(tokenids) * 100:.1f}%)")
 
    return kept

def buildnegativetable(counts, tablesize=negativetablesize):
    # negatives should be sampled proportional to freq^0.75 rather than uniformly
    # uniform sampling underrepresents common words as negatives relative to
    # how often they show up as "accidental" context, which is probably why
    # everything was just coming up as similar at one point
 
    freq = counts.astype(cp.float64) ** 0.75
    freq = freq / freq.sum()
    cumulative = cp.cumsum(cp.array(freq))
 
    positions = cp.arange(tablesize) / tablesize
    table = cp.searchsorted(cumulative, positions).astype(cp.int32)
 
    return table

def embedding(corpus):
    # i've added comments and steps to show my process of thinking but i have also written up documents on
    # my original planning and stuff, will put in the github or possibly a website

    # i did about 2 weeks worth of research and planning for bgpt 3.0, so i would like to
    # thank these research papers and websites i have learned and pulled techinques from:

    # Mikolov et al., 2013. Efficient Estimation of Word Representations in Vector Space
    # Vaswani et al., 2017. Attention Is All You Need
    # Wood et al., 2023. NLP’s quantum leap: from N-grams to GPT-4
    # Cabrera et al., 2017. Transformer Explainer: LLM Transformer Model Visually Explained

    # without these papers, blogs and websites, this would not be possible.

    #step 1: tokenise and subsample everything
    print("Tokenising data...")

    wordtoid, idtoword, tokens, counts = tokenize(corpus)

    n = len(wordtoid)

    print("Subsampling frequent words...")
    tokens = subsample(tokens, counts)

    #step 2: create word vectors at random weights
    
    matrix = (cp.random.randn(n, dimensions) * matrixscalar).astype(cp.float32) # float32 is a big perf boost

    #step 3: prepare for pair lookups
    windowsize = 5

    tokens = cp.asarray(tokens, dtype=cp.int32)
    tokenslen = len(tokens)

    batchesperepoch = (tokenslen * 2 * windowsize) // batchsize   #same number of pairs as before
    usedpairs = batchesperepoch * batchsize
    print(f"Tokens: {tokenslen}, pairs per epoch: {usedpairs}, batches per epoch: {batchesperepoch}")

    #step 4: output weight matrix
    outputmatrix = (cp.random.randn(n, dimensions) * matrixscalar).astype(cp.float32)


    #step 4.5: negative sampling table
    print("Building negative sampling table...")
    negativetable = buildnegativetable(counts)
    tablesize = len(negativetable)

    #step 5 : training loop (if this goes wrong ill cry)
    groupsize = 250
    groupnegatives = 20

    epoch = 0
    improvement = 999 #set to something that will never be below the threshold lol
    while improvement > epochthresh:
        epoch += 1
        losstotal = 0

        for i in range(batchesperepoch):
            pos = cp.random.randint(0, tokenslen, size=batchsize)
            offset = cp.random.randint(1, windowsize + 1, size=batchsize)
            offset = cp.where(cp.random.rand(batchsize) < 0.5, -offset, offset)
            ctx = pos + offset
            ctx = cp.where((ctx < 0) | (ctx >= tokenslen), pos - offset, ctx)   # flip direction at the corpus edges

            inputs = tokens[pos]
            targets = tokens[ctx]
            batchlen = batchsize

            negatives = negativetable[cp.random.randint(0, tablesize, size=(batchlen // groupsize, groupnegatives))]

            # FORWARD PASS
            vector = matrix[inputs]
            v = vector.reshape(-1, groupsize, dimensions)  
            positivevector = outputmatrix[targets]
            negativevector = outputmatrix[negatives]

            positivescore = cp.sum(vector * positivevector, axis=1)
            negativescore = cp.matmul(v, negativevector.transpose(0, 2, 1))

            positivesig = 1 / (1 + cp.exp(-positivescore))
            negativesig = 1 / (1 + cp.exp(negativescore))

            loss = -cp.log(positivesig + 1e-9) - cp.log(negativesig + 1e-9).sum(axis=2).reshape(-1) #add tiny amount to prevent log(0)
            losstotal += loss.sum()

            # BACKWARD PASS
            gradpositive = (positivesig - 1)
            gradnegative = (1 - negativesig)

            gradvector = gradpositive[:, None] * positivevector + cp.matmul(gradnegative, negativevector).reshape(-1, dimensions)
            gradpositivetarget = gradpositive[:, None] * vector
            gradnegativetarget = cp.matmul(gradnegative.transpose(0, 2, 1), v)

            cupyx.scatter_add(matrix, inputs, -learningrate * gradvector)
            cupyx.scatter_add(outputmatrix, targets, -learningrate * gradpositivetarget)
            cupyx.scatter_add(outputmatrix, negatives.reshape(-1), -learningrate * gradnegativetarget.reshape(-1, dimensions))

        if epoch % 1 == 0:
            avgloss = float(losstotal) / usedpairs
            print(f"Epoch: {epoch}")
            try: # this is really lazy but icba rn
                improvement = lastloss - avgloss
                print(f"Loss improvement: {(lastloss - avgloss):.6f}")
            except NameError:
                pass

            print(f"Avg loss: {avgloss:.6f}\n")
            lastloss = avgloss

    print("Negligible loss improvement detected, stopping training...")

    matrix = cp.asnumpy(matrix) # i switched to cupy last minute because i forgot so ts kinda lazy

    return matrix, wordtoid, idtoword


def generatetfidf():
    global vocab, idf, tfidfvectors

    vocab, idf, tfidfvectors = tfidf.buildtfidf(shared.userquestions) #only makes vectors for userquestions
