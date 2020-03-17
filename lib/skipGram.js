class SkipGram {

    constructor(voc_size, hidden_nodes) {
        this.n = voc_size;
        this.hidden_nodes = hidden_nodes;
        this.activationFunction = {
            softmax: vector => {
                vector = vector.data
                let sum = 0;
                for(let i of vector) {
                    sum += Math.pow(Math.E, i);
                }
                if(sum == Infinity || sum == NaN) 
                    sum = 100;
                let result = [];
                for(let i = 0; i < vector.length; i++) {
                    result.push(Math.pow(Math.E, vector[i]) / sum);
                }
                return result;
            }
        }
        this.lossFunction = {
            loss: (vec1, vec2) => {
                if(vec1.length != vec2.length)
                    throw new Error("Dimensions of vectors do not match!");
                let sum = 0;
                for(let i = 0; i < vec1.length; i++) {
                    sum += vec1[i] - vec2[i];
                }
                return sum;
            }
        }
        this.weights_ih = new Matrix(this.n, this.hidden_nodes);
        this.weights_ho = new Matrix(this.hidden_nodes, this.n);
        this.weights_ih.randomize();
        this.weights_ho.randomize();
    }

    predict(vector) {
        // input layer
        vector = Matrix.fromArray(vector);
        vector = Matrix.transpose(vector);
        // hidden layer
        let hidden = Matrix.multiply(vector, this.weights_ih);
        // output layer 
        let output = Matrix.multiply(hidden, this.weights_ho);
        output = Matrix.transpose(output);
        return this.activationFunction.softmax(output); // output is an array 
    }

    // * does not work *
    train2(x_, t, lr) {

        // (1) calculate the network's prediction for training example x
        let x = Matrix.fromArray(x_);
        x = Matrix.transpose(x);
        let h = Matrix.multiply(x, this.weights_ih);
        let z = Matrix.multiply(h, this.weights_ho);
        z = Matrix.transpose(z);
        let y = this.activationFunction.softmax(z);
        let vec_y = Matrix.fromArray(y);


        // (3) calculate the derivatives and update them (multipy with lr)
        let t_ = Matrix.fromArray(t);

        // W'
        // let d_w_ho = Matrix.subtract(vec_y, t_).multiply(h);
        for(let j = 0; j < this.n; j++) {
            let zw = (y[j] - t[j]);
            for(let i = 0; i < this.hidden_nodes; i++) {
                let h_i = h.data[0][i];
                console.log(lr * zw * h_i);
                this.weights_ho.data[i][j] += lr * zw * h_i;
            }
        }

        // W
        for(let j = 0; j < this.hidden_nodes; j++) {
            let sum = 0;
            for(let k = 0; k < this.n; k++) {
                sum += (y[k] - t[k]) * this.weights_ho.data[j][k];
            }
            for(let i = 0; i < this.n; i++) {
                this.weights_ih.data[i][j] += lr * x_[i] * sum;
            }
        }

    }
 
    // * does not work *
    train3(input_array, output_array) {

        let lr = 0.9;
        let targets = Matrix.fromArray(output_array);

        // (1) calculate (feed-forward)
        let x = Matrix.fromArray(input_array);
        x = Matrix.transpose(x);
        let h = Matrix.multiply(x, this.weights_ih);
        let z = Matrix.multiply(h, this.weights_ho);
        z = Matrix.transpose(z);
        let y = this.activationFunction.softmax(z);
        let vec_y = Matrix.fromArray(y);

        // (2) loss, calc gradients, update weights
        let output_errors = Matrix.subtract(vec_y, targets);

        // W' (Hidden - Output)
        let gradients = vec_y.multiply(output_errors);
        gradients.multiply(lr);
        console.log(gradients);
        let h_t = Matrix.transpose(h);
        console.log(h);
        let deltas_ho = gradients.multiply(h);
        console.log(deltas_ho);
        this.weights_ho.add(deltas_ho);

        // W (Input - Hidden)
        let weights_ho_t = Matrix.transpose(this.weights_ho);
        let hidden_errors = weights_ho_t.multiply(output_errors);
        let gradients_hidden = h.multiply(hidden_errors);
        gradients_hidden.multiply(lr);
        let input_t = Matrix.transpose(x);
        let deltas_ih = gradients_hidden.multiply(input_t);
        this.weights_ih.add(deltas_ih);


    }

    train(x_, t, lr = 0.9, epochs = 1000) {


        // (1) the network's prediction for training example x
        let x = Matrix.fromArray(x_);
        x = Matrix.transpose(x);
        t = Matrix.fromArray(t);
        console.log(t);

        for(let epoch = 1; epoch <= epochs; epoch++) {

            let h = Matrix.multiply(x, this.weights_ih);
            let z = Matrix.multiply(h, this.weights_ho);
            z = Matrix.transpose(z);
            let y = Matrix.fromArray(this.activationFunction.softmax(z));

            // (2) calculate gradients and update weights

            // W' (Hidden - Output)
            let subtraction = Matrix.subtract(y, t);
            let product = Matrix.product(subtraction, h);
            let gradients = Matrix.transpose(product);
            gradients.multiply(lr);
            this.weights_ho.add(gradients);

            // W (Input - Hidden)
            let sub = Matrix.subtract(y, t);
            let brackets = Matrix.multiply(this.weights_ho, sub)
            brackets = Matrix.transpose(brackets);
            let gradients_hidden = Matrix.product(Matrix.transpose(x), brackets);
            gradients_hidden.multiply(lr);
            this.weights_ih.add(gradients_hidden);

            // log sth to console
            console.log("epoch " + epoch + "/" + epochs);

        }

    }

}

//  - calculate derivatives for every i,j pair
//  - does feed forward work?
//  - make training more efficient (epochs within network!)
//  - make improved version of it!
//  - we are not using the activation function!
