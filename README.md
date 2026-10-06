<h2>
Word2vec style vector embedding to understand meaning between words in .txt files.<br>
Will be used to develop badGPT 3.0, which will feature full transformer architecture
</head>
<br><br>
</h2>


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

<h1>Tests on 500mb + window size 5</h1>
<u><b>(RTX 3080 10GB + 16GB 3600mhz RAM w/ 500MB .txt file)</b></u><br><br>
<h3> @ <b>50</b> dims:</h2>
<img src='Figure_1.png'>
<i>
Figure 1 shows relatively similar projected vector magnitude between gender of words<br>
</i>
<br>


<h3>@ <b>600</b> dims:</h3>
<img src='Figure_2.png'>
<i>
Figure 2 shows more precise precise vector magnitude<br>
</i>
<br>
<b>
<h3>
From these tests, we can prove that: <br><br>
<u>
Precision(<i>N</i>) = <i>k</i> &middot; <i>N<sub>dim</sub></i> where K is a positive constant
</u>
</h3>
</b>

<br><br><br>
<h2>All of this code was 100% human-written :)</h2>
