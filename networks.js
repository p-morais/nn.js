var layers = require('./layers')

module.exports = {BinaryMLP}

BinaryMLP = function(in_dim, hidden_units) {
    //TODO: write initializer functions for weights
    self.in = layers.Linear(null, null)
    self.in.add_layer(layers.ReLu())
    self.in.add_layer(layers.Linear(null, null))
    self.in.add_layer(layers.SoftmaxCrossEntropy())

    self.out = self.in.get_last()
}

BinaryMLP.prototype.predict = function(x) {
    return this.in.forward(x)
}

BinaryMLP.prototype.fit =
    function(x_batch,
             y_batch,
             eta,
             momentum,
             decay) {

    this.predict(x_batch)
    return this.out.backward(y_batch, eta, momentum, decay)
}
