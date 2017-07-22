nj = require('numjs')
module.exports = {
    //TODO: more numerically stable implementation.
    //also dear JS: implement operator overloading, this sucks
    sigmoid: function(x) {
        x_out = nj.multiply(x, -1)
        x_out = nj.add(nj.exp(x_out), 1)
        x_out = nj.divide(nj.ones(x_out.shape), x_out)
        return x_out
    }
}
