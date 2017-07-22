var nj = require('numjs')
var util = require('./util')

module.exports = {Linear, ReLu, SoftmaxCrossEntropy}

Layer = function() {
    this.input = null
    this.output = null

    this.prev_layer = null
    this.next_layer = null
}

Layer.prototype.get_last = function() {
    last_layer = this

    while(last_layer.next_layer != null)
        last_layer = last_layer.next_layer

    return last_layer
}

Layer.prototype.add_layer = function(layer) {
    last_layer = this.get_last()

    last_layer.next_layer = layer
    last_layer.next_layer.prev_layer = last_layer
}


// Currently just passing through gradients with updates. Might switch to a
// gradient buffer and a seperate optimizer class which updates weights based
// on each layer's gradient buffer, similar to pytorch but with no autodiff ):
/*============================================================================*/
Linear = function(W, b) {
    Layer.call(this)

    this.dW = 0
    this.W = W
    this.b = b
}

Linear.prototype.forward = function(x) {
    this.input = x
    this.output = nj.add(nj.dot(x, this.W), this.b)

    if(this.next_layer != null)
        return this.next_layer.forward(this.output)
    else
        return this.output
}

Linear.prototype.backward = function(delta, eta, momentum, decay) {
    dJdW = nj.dot(this.input.T, delta)
    dJdW = nj.clip(dJdW, -50, 50)

    this.dW = nj.add(dJdW, nj.multiply(this.W, decay))
    this.dW = nj.multiply(this.dW, eta)
    this.dW = nj.subtract(this.dW, nj.multiply(this.dW, momentum))

    this.W = nj.subtract(this.W, this.dW)
    this.b = nj.subtract(this.b, nj.sum(delta, 0))
    this.b = nj.multiply(this.b, nj.sum(delta, 0))

    grad = nj.dot(delta, this.W.transpose())  # dz/da
    if(this.prev_layer != null)
        this.prev_layer.backward(grad, eta, momentum, decay)
}


/*============================================================================*/
ReLu = function() {
    Layer.call(this)
}

ReLu.prototype.forward = function() {
    this.input = x
    this.output = nj.zeros(x.shape)

    for(var i = 0; i < this.input.shape[0]; i++)
        if(this.input.get(i) > 0)
            this.output.set(i, this.input.get(i))

    if(this.next_layer != null)
        return this.next_layer.forward(this.output)
    else
        return this.output
}

ReLu.prototype.backward = function(err, eta, momentum, decay) {
    grad = nj.zeros(this.input.shape)
    for(var i = 0; i < this.input.shape[0]; i++)
        if(this.input.get(i) > 0)
            grad.set(i, this.input.get(i))

    if(this.prev_layer != null) {
        delta = nj.multiply(grad, err)
        this.prev_layer.backward(delta, eta, momentum, decay)
    }
}


/*============================================================================*/
SoftmaxCrossEntropy = function() {
    Layer.call(this)
}

SoftmaxCrossEntropy.prototype.forward = function(x) {
    this.input = x
    this.output = util.sigmoid(this.input)
}

SoftmaxCrossEntropy.prototype.loss = function(y) {
    yHat = nj.add(this.output, 1e-10)
    yHat_neg = nj.multiply(yHat, -1)

    y_neg = nj.multiply(y, -1)

    e1 = nj.multiply(y_neg, nj.log(yHat))

    e2_t1 = nj.multiply(nj.subtract(nj.ones(y.shape), y)), -1)
    e2_t2 = nj.log(nj.subtract(nj.ones(yHat.shape), yHat))
    e2 = nj.multiply(e2_t1, e2_t2)

    return nj.mean(nj.multiply(e1, e2))
}

SoftmaxCrossEntropy.prototype.backward = function(t, eta, momentum, decay) {
    cost = nj.subtract(self.output, t)
    if(this.prev_layer != null)
        this.prev_layer.backward(cost, eta, momentum, decay)

    return self.loss(t)
}
