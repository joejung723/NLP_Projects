# TF x IDF Summarizer Using Cosine Similarity

Command Line
------------
tfidf_summarizer.py text_file


Details & Issues
----------------
Program can be executed through command line and will output a file that summarizes the text_file according to the top
scores of the sentences in that text_file.  By using the TF x IDF score and weights, program will take the average TF x IDF
score of the sentences and write into the output file the top ranked sentences.
Some issues that were encountered when handling a text file were anything that wasn't a word.  The text file had to be
a clean text file.
Issues arose when certain characters and citations were making certain sentences score higher than normal sentences because
of the unique characters/words.  I tried to handle these issues by adjusting the values in which sentences were chosen by
assigning a range above the average TF x IDF score.


Author
------
Joe Jung
