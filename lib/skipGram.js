class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let softmax = new ActivationFunction(
    (x) => {
       x = x.toArray();
       let sum = 0;
       for(let i = 0; i < x.length; i++) {
           sum += Math.exp(Math.abs(x[i]));
       }
       let out = [];
       for(let i = 0; i < x.length; i++) {
           out.push(Math.exp(x[i]) / sum);
       }
       return Matrix.fromArray(x);
    },
    (vec) => {
        function softmax(x) {
            x = x.toArray();
            let sum = 0;
            for(let i = 0; i < x.length; i++) {
                sum += Math.exp(x[i]);
            }
            let out = [];
            for(let i = 0; i < x.length; i++) {
                out.push(Math.exp(x[i]) / sum);
            }
            return Matrix.fromArray(out);
        }
        let one_minus = softmax(vec).map(e => 1 - e);
        return softmax(vec).multiply(one_minus);
    }
);


class SkipGram {

  constructor(in_nodes, hid_nodes = 100, learningRate = 0.9) {
    this.input_nodes = in_nodes;
    this.hidden_nodes = hid_nodes;
    this.output_nodes = in_nodes;

    this.weights_ih = new Matrix(this.input_nodes, this.hidden_nodes);
    this.weights_ho = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    // TODO: copy these as well
    this.setLearningRate(learningRate);
    this.setActivationFunction();

  }

  predict(input_array) {
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(Matrix.transpose(inputs), this.weights_ih);

    // Generating the output's output!
    let output = Matrix.multiply(hidden, this.weights_ho);
    // output.map(this.activation_function.func);
    output = this.activation_function.func(output);

    // Sending back to the caller!
    return output.toArray();
  }

  setLearningRate(learning_rate = 0.1) {
    this.learning_rate = learning_rate;
  }

  setActivationFunction(func = softmax) {
    this.activation_function = func;
  }

  train(input_array, target_array) {
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(Matrix.transpose(inputs), this.weights_ih);

    // Generating the output's output!
    let outputs = Matrix.multiply(hidden, this.weights_ho);
    outputs = this.activation_function.func(outputs);

    // Convert array to matrix object
    let targets = Matrix.fromArray(target_array);

    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, outputs);

    // let gradient = outputs * (1 - outputs);
    // Calculate gradient
    let gradients = this.activation_function.dfunc(outputs);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    // Calculate deltas
    let weight_ho_deltas = Matrix.multiply(gradients, hidden);

    // Adjust the weights by deltas
    this.weights_ho.add(Matrix.transpose(weight_ho_deltas));

    // Calculate the hidden layer errors
    let hidden_errors = Matrix.multiply(this.weights_ho, output_errors);

    // Calculate hidden gradient
    let hidden_gradient = this.activation_function.dfunc(hidden);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    // Calcuate input->hidden deltas
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

    this.weights_ih.add(Matrix.transpose(weight_ih_deltas));

  }

  getWordVector(row) {
    return this.weights_ih.data[row];
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == 'string') {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes);
    nn.weights_ih = Matrix.deserialize(data.weights_ih);
    nn.weights_ho = Matrix.deserialize(data.weights_ho);
    nn.learning_rate = data.learning_rate;
    return nn;
  }

}