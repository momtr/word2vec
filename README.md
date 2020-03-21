# word2vec
A JavaScript implementation of word2vec (Google's SkipGram).
I know there are much faster libraries for other languages, however, it was a nice project to understand the concept of word2vec and skip grams.

## Concept
The basic concept of a skip gram is that we take a context word and predict several target words. That means we collect all neighboring words (within a window-size) (Y) of a certain word (X). We train a model to solve:

> SkipGram(X) = [Y1, Y2, Y3, ..., Yn]

## Code
<br>
Skip grams are neural networks. Parts of the code for the neural network can be found here: https://github.com/CodingTrain/Toy-Neural-Network-JS (by Coding Train). 
