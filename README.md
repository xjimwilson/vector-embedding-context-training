<h3>
Word2vec style vector embedding to understand meaning between words in txt files.<br>
Will be used to develop badGPT 3.0, which will feature full transformer architecture
</head>
<br><br>
</h3>
<u><b>Tested on RTX 3080 10GB with 16GB RAM with a 500MB file, may improve performance and test more in the future.</b></u>

<br>
<h1><b><u>HOW TO USE (first time):</u></b></h1>
<h3>
0.5. install dependencies obviously<br>
1. run main.py, which features a cli with instructions.<br>
2. train on the sample file provided (stories.txt)<br>
3. should take a while depending on your hardware, but will create a .npz file<br>
4. press m to manually test the fully trained model<br>
</h3>
(Simply drag and drop txt files in datasets then train)
<br>
<h3> !! G to generate is not currently supported, placeholder for ai generation in the future !! </h3>
<br><br>

Model trained on 500mb @ 600 dims + window size 5:
<img src='Figure_1.png'>
Figure 1 shows similar projected vector magnitude between gender of words<br>
<br>
<h2>All of this code was 100% human-written :)</h2>

