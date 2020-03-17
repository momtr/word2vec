class Word2Vec {

    constructor(text, hiddenNodes, stopWords) {
        // get array of sentences and arrays of words 
        // let sentences = text.split(/[^\.!\?]+[\.!\?]+/g);
        let sentences = text.split(". ");
        this.data = [];
        // n is the number of words
        this.n = 0;
        for(let i of sentences) {
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
            hiddenNodes: hiddenNodes
        }
        this.brain = new NeuralNetwork(this.n, this.hyperparameters.hiddenNodes, this.n);
    }

    oneHotVector(length, zeroIndex) {
        let vec = [];
        for(let i = 0; i < length; i++) {
            vec.push(0);
        }
        vec[zeroIndex] = 1;
        return vec;
    }

    train(windowSize, epochs) {
        console.log("started training")
        // get the training pairs
        // context and target words 
        let training_pairs = [];
        for(let i of this.data) {
            for(let j = 0; j < i.length; j++) {
                for(let k = j - windowSize; k < j + windowSize; k++) {
                    if(j != k && k >= 0 && k < i.length) {
                        training_pairs.push([i[j], i[k]]);
                    }
                }
            }
        }
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
        if(typeof a === "string" && typeof b === "string") {
            a = this.brain.weights.data[this.lookup[a]];
            b = this.brain.weights.data[this.lookup[b]];
        }
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
        return sum / (abs_a * abs_b); 2
    }

    cosine_distances(word) {
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

}

// problems:
//  - activation funcions
//  - removing stop words (is)
//  - I'm => I am
//  - split on !?
//  - increase number of hidden nodes
//  - in lookup table only store indexes of one position in one-hot encoded vector
//  - text from different sources
//  - if word exists already
//  - plural

//  - activation function 
//  - activation function in word2vec.js
