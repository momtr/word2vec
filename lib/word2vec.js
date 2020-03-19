class Word2Vec {

    constructor(text, hiddenNodes, stopWords, learningRate) {
        if(!stopWords)
            stopWords = ["me", "and", "i", "is", "or", "the", "of", "in", "on", "off", "these", "those", "that", "were", "was", "they", "you", "she", "he", "soon", "but", "as", "however", "already", "a", "an", "to", "got", "very", "herself", "himself", "’"];
        // get array of sentences and arrays of words 
        // let sentences = text.split(/[^\.!\?]+[\.!\?]+/g);
        text = text.replace("!", ".");
        text = text.replace("?", ".");
        let sentences = text.split(". ");
        this.data = [];
        // n is the number of words
        this.n = 0;
        for(let i of sentences) {
            i = i.replace(",", "");
            i = i.replace("\"", "");
            i = i.replace("'", "");
            i = i.replace(":", "");
            i = i.replace(";", "");
            i = i.replace("´", "");
            let words = i.split(" ");
            // "lowercase" them
            let arr = [];
            for(let k = 0; k < words.length; k++) {
                words[k] = words[k].toLowerCase();
                if(stopWords.indexOf(words[k]) === -1) {
                    arr.push(words[k]);
                    this.n++;
                } 
            }
            this.data.push(arr); 
        }
        // the lookup table for one-hot encoded vectors
        this.lookup = {};
        this.lookup_vec2word = {};
        // go through all words
        let wordIndex = 0;
        for(let i = 0; i < this.data.length; i++) {
            for(let j = 0; j < this.data[i].length; j++) {
                this.lookup[this.data[i][j]] = wordIndex;
                this.lookup_vec2word[wordIndex] = this.data[i][j]
                wordIndex++;
            }
        }
        // one-hot encode
        // the network's brain (Neural Network)
        this.hyperparameters = {
            hiddenNodes: hiddenNodes,
            learningRate: learningRate
        }
        this.brain = new SkipGram(this.n, this.hyperparameters.hiddenNodes, this.hyperparameters.learningRate);
    }

    oneHotVector(length, zeroIndex) {
        let vec = [];
        for(let i = 0; i < length; i++) {
            vec.push(0);
        }
        if(zeroIndex == 0 || zeroIndex)
            vec[zeroIndex] = 1;
        return vec;
    }

    train2(windowSize = 5, epochs = 70, download) {
        console.log("started training")
        // get the training pairs
        // context and target words 
        let training_pairs = [];
        for(let i of this.data) {
            for(let j = 0; j < i.length; j++) {
                // two loops, so that network is trained on index + 1 the most (and not on index + winSize - 1)
                for(let k = j - windowSize; k < j; k++) {
                    if(k >= 0) 
                        training_pairs.push([i[j], i[k]]);
                }
                for(let k = j + windowSize - 1; k > j; k--) {
                    if(k < i.length) {
                        training_pairs.push([i[j], i[k]]);
                    }
                }
            }
        }
        console.log(training_pairs);
        // train the network
        for(let epoch = 0; epoch < epochs; epoch++) {
            console.log(epoch);
            for(let i of training_pairs) {
                let context = this.oneHotVector(this.n, this.lookup[i[0]]);
                let target = this.oneHotVector(this.n, this.lookup[i[1]]);
                this.brain.train(context, target);
            }
        }
        console.log("finished training");
        if(download)
            saveJSON(this, 'model.json');
    }

    train(windSize = 5, epochs = 70, download) {
        let examples = [];
        // got through all sentences
        for(let i of this.data) {
            // go through each sentence
            for(let j = 0; j < i.length; j++) {
                // go through all words
                let x = this.oneHotVector(this.n, this.lookup[i[j]]);
                let targets = this.oneHotVector(this.n);
                for(let k = j - windSize; k <= j + windSize; k++) {
                    if(j != k && k >= 0 && k < i.length) {
                        targets[this.lookup[i[k]]] = 1;
                    }
                }
                examples.push({ x: x, y: targets })
            }
        }
        console.log(examples);
        // train
        for(let epoch = 1; epoch <= epochs; epoch++) {
            console.log(epoch);
            for(let i of examples) {
                let context = i.x;
                let target = i.y;
                this.brain.train(context, target);
            }
        }
        console.log("finished training");
        if(download)
            saveJSON(this, 'model.json');
    }

    predict(vector) {
        // vector can be:
        //  - number: index of vector where entry is one
        //  - array: one-hot encoded vector
        //  - string: the actual word
        if(!isNaN(vector)) {
            vector = this.oneHotVector(this.n, vector);
        } else if(typeof vector == 'string') {
            vector = this.oneHotVector(this.n, this.lookup[vector.toLocaleLowerCase()]);
        } else if(typeof vector == 'object') {
            if(vector.length != this.n)
                throw new Error("Dimension of vector to predict is " + vector.length + " and does not match with dimension " + this.n + ". Use the number of all words as the dimension");
        }
        // predict with NeuralNetwork.predict
        let prediction = this.brain.predict(vector);
        // apply the softmax function to the prediction
        function softmax(vector) {
            let sum = 0;
            for(let i of vector) {
                sum += Math.pow(Math.E, i);
            }
            // new vector
            let result = [];
            for(let i = 0; i < vector.length; i++) {
                result.push(Math.pow(Math.E, vector[i]) / sum);
            }
            return result;
        }
        // let softmaxAppliedVector = softmax(prediction);
        let softmaxAppliedVector = prediction;
        // get the corresponding words
        let mapping = [];
        for(let i = 0; i < softmaxAppliedVector.length; i++) {
            mapping.push({
                word: this.lookup_vec2word[i],
                score: softmaxAppliedVector[i]
            })
        }
        // sort the array
        mapping.sort((a, b) => {
            if(a.score < b.score)
                return 1;
            else 
                return -1;
        })
        return mapping;
    }

    // create Word2Vec object from json file
    static load(json, text, stopWords) {
        let word2vec = new Word2Vec(text, json.hyperparameters.hiddenNodes, stopWords);
        word2vec.lookup = json.lookup;
        word2vec.lookup_vec2word = json.lookup_vec2word;
        word2vec.n = json.n;
        word2vec.brain = NeuralNetwork.deserialize(json.brain);
        return word2vec;
    }

    cosine_distance(a, b) {
        if(typeof a === "string") 
            a = this.brain.getWordVector(this.lookup[a]);
        else
            a = a.toArray();
        if(typeof b === "string")
            b = this.brain.getWordVector(this.lookup[b]);
        else
            b = b.toArray();
        let sum = 0;
        for(let i = 0; i < a.length && i < b.length; i++) {
            sum += a[i] * b[i];
        };
        let abs_a = 0; 
        for(let i = 0; i < a.length; i++) {
            abs_a += a[i] * a[i];
        }
        abs_a = Math.sqrt(abs_a, 2);
        let abs_b = 0; 
        for(let i = 0; i < b.length; i++) {
            abs_b += b[i] * b[i];
        }
        abs_b = Math.sqrt(abs_b, 2);
        return sum / (abs_a * abs_b);
    }

    cosine_distances(word) {
        if(typeof word === 'string')
            word = this.vec(word);
        let map = [];
        let words = Object.keys(this.lookup);
        for(let i = 0; i < words.length; i++) {
            map.push({
                word: words[i],
                score: this.cosine_distance(word, words[i])
            });
        }
        // sort map
        map.sort((a, b) => {
            if(a.score < b.score)
                return 1;
            else    
                return -1;
        });
        return map;
    }

    // word -> vec
    vec(word) {
        if(!word)
            throw new Error("Word must be defined!");
        return Matrix.fromArray(this.brain.getWordVector(this.lookup[word]));
    }

    // vec -> word
    toWord(vec) {
        return this.cosine_distances(vec);
    }

    subtract(word1, word2) {
        return Matrix.subtract(this.vec(word1), this.vec(word2));
    }

    add(word1, word2) {
        return this.vec(word1).add(this.word(word2));
    }

    save(fileName = ('word2vec_model_' + Date.now())) {
        saveJSON(this, fileName);
    }

}

// problems:
//  - removing stop words (is)
//  - I'm => I am
//  - some words are in the list twice
//  - split on !?
//  - increase number of hidden nodes
//  - in lookup table only store indexes of one position in one-hot encoded vector
//  - text from different sources
//  - if word exists already
//  - plural
//  - IMPROVED VERSION (SEE GIOOGLES PAPER ON SKIP GRAMS!)
