const NeuralNetwork = (function() {

    class NeuralNetwork {
        constructor(inputNodes, layers, learningRate) {
            this.inputNodes = inputNodes || 1;
            this.layers = layers;
            this.learningRate = learningRate || 0.1;

            for(let i = 0; i < this.layers.length; i++) {
                this.layers[i].weights = new Matrix(this.layers[i].nodes, i === 0 ? this.inputNodes : this.layers[i - 1].nodes);
                this.layers[i].bias = new Matrix(this.layers[i].nodes, 1);

                this.layers[i].weights.random(-1, 1);
                this.layers[i].bias.random(-1, 1);
            }
        }

        predict(inputs) {
            inputs = Matrix.from_arr(inputs);

            let prevLayerValues = inputs;

            for(let layer of this.layers) {
                layer.layerValues = Matrix.multiply(layer.weights, prevLayerValues);
                layer.layerValues.add(layer.bias);
                layer.layerValues.map(layer.activationFunction.func);

                prevLayerValues = layer.layerValues;
            }

            const outputs = prevLayerValues;
            return Matrix.to_arr(outputs);
        }

        train(inputArray, targetArray) {
            const inputs = Matrix.from_arr(inputArray);

            let prevLayerValues = inputs;

            for(let layer of this.layers) {
                layer.layerValues = Matrix.multiply(layer.weights, prevLayerValues);
                layer.layerValues.add(layer.bias);
                layer.layerValues.map(layer.activationFunction.func);

                prevLayerValues = layer.layerValues;
            }

            const outputs = this.layers[this.layers.length - 1].layerValues;
            const targets = Matrix.from_arr(targetArray);

            let prevError = null;

            for(let i = this.layers.length - 1; i >= 0; i--) {
                const layer = this.layers[i];

                let error;

                if(!prevError) {
                    error = Matrix.substract(targets, outputs);
                } else {
                    const nextWeights = this.layers[i + 1].weights;
                    const nextWeightsTransposed = Matrix.transpose(nextWeights);

                    error = Matrix.multiply(nextWeightsTransposed, prevError);
                }

                const gradient = Matrix.map(layer.layerValues, layer.activationFunction.dfunc);
                gradient.multiply(error);
                gradient.multiply(this.learningRate);

                const prevValues = i === 0 ? inputs : this.layers[i - 1].layerValues;
                const prevValuesTransposed = Matrix.transpose(prevValues);
                const layerWeightsDeltas = Matrix.multiply(gradient, prevValuesTransposed);

                layer.weights.add(layerWeightsDeltas);
                layer.bias.add(gradient);

                prevError = error;
            }
        }

        setLearningRate(learningRate) {
            this.learningRate = learningRate;
        }

        serialize() {
            return JSON.stringify(this);
        }

        static deserialize(data) {
            if(typeof data === 'string') {
                data = JSON.parse(data);
            }
            let layers = [];

            for(let layer of data.layers) {
                layers[i] = Layer.deserialize(layer);
            }

            const neuralnetwork = NeuralNetwork.createNeuralNetwork({
                inputNodes: data.inputNodes,
                layers: layers,
                learningRate: data.learningRate
            });

            return neuralnetwork;
        }
    }

    class Layer {
        constructor(nodes, activationFunction) {
            this.nodes = nodes;
            this.activationFunction = activationFunction;
            this.weights = null;
            this.bias = null;
            this.layerValues = null;
        }

        static deserialize(data) {
            if(typeof data === 'string') {
                data = JSON.parse(data);
            }

            const layer = NeuralNetwork.createLayer({
                nodes: data.nodes,
                activation: data.activation
            });
            layer.weights = data.weights;
            layer.bias = data.bias;
            layer.layerValues = layerValues;

            return layer;
        }
    }

    function createNeuralNetwork(config) {
        return new NeuralNetwork(config.inputNodes, config.layers, config.learningRate);
    }

    function createLayer(config) {
        const activation = {};

        switch(config.activation) {
            case 'sigmoid':
                activation.func = sigmoid;
                activation.dfunc = dsigmoid;
            break;
            default:
                activation.func = sigmoid;
                activation.dfunc = dsigmoid;
        }

        return new Layer(config.nodes, activation);
    }

    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    function dsigmoid(y) {
        return y * (1 - y);
    }

    return {
        createNeuralNetwork,
        createLayer
    };

})();

class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;

        this.matrix = [];

        for(let i = 0; i < this.rows; i++) {
            this.matrix[i] = [];
            for(let j = 0; j < this.cols; j++) {
                this.matrix[i][j] = 0;
            }
        }
    }

    random(min, max) {
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.cols; j++) {
                this.matrix[i][j] = Math.random() * (max - min) + min;
            }
        }
    }

    print() {
        console.table(this.matrix);
    }

    map(f) {
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.cols; j++) {
                let val = this.matrix[i][j];
                this.matrix[i][j] = f(val, i, j);
            }
        }
    }

    add(n) {
        if(n instanceof Matrix) {
            if(this.rows != n.rows || this.cols != n.cols) {
                throw 'Matrices are not the same size';
            }

            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.cols; j++) {
                    this.matrix[i][j] += n.matrix[i][j];
                }
            }
        } else {
            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.cols; j++) {
                    this.matrix[i][j] += n;
                }
            }
        } 
    }

    substract(n) {
        if(n instanceof Matrix) {
            if(this.rows != n.rows || this.cols != n.cols) {
                throw 'Matrices are not the same size';
            }

            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.cols; j++) {
                    this.matrix[i][j] -= n.matrix[i][j];
                }
            }
        } else {
            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.cols; j++) {
                    this.matrix[i][j] -= n;
                }
            }
        } 
    }

    multiply(n) {
        if(n instanceof Matrix) {  
            if(this.rows !== n.rows || this.cols !== n.cols) {
                throw "these matrices have to be the same size";
            }

            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.cols; j++) {
                    this.matrix[i][j] *= n.matrix[i][j];
                }
            }
        } else {

            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.cols; j++) {
                    this.matrix[i][j] *= n;
                }
            }
        }
    }

    static transpose(m) {
        let result = new Matrix(m.cols, m.rows);

        for(let i = 0; i < result.rows; i++) {
            for(let j = 0; j < result.cols; j++) {
                result.matrix[i][j] = m.matrix[j][i];
            }
        }

        return result;
    }

    static add(m, n) {
        let result = new Matrix(m.rows, m.cols);

        if(n instanceof Matrix) {
            if(m.rows !== n.rows || m.cols !== n.cols) {
                throw 'Matrices are not the same size';
            }

            for(let i = 0; i < result.rows; i++) {
                for(let j = 0; j < result.cols; j++) {
                    result.matrix[i][j] = m.matrix[i][j] + n.matrix[i][j];
                }
            }

            return result;
        } else {
            for(let i = 0; i < result.rows; i++) {
                for(let j = 0; j < result.cols; j++) {
                    result.matrix[i][j] = m.matrix[i][j] + n;
                }
            }
        } 

        return result;
    }

    static substract(m, n) {
        let result = new Matrix(m.rows, m.cols);

        if(n instanceof Matrix) {
            if(m.rows !== n.rows || m.cols !== n.cols) {
                throw 'Matrices are not the same size';
            }

            for(let i = 0; i < result.rows; i++) {
                for(let j = 0; j < result.cols; j++) {
                    result.matrix[i][j] = m.matrix[i][j] - n.matrix[i][j];
                }
            }

            return result;
        } else {
            for(let i = 0; i < result.rows; i++) {
                for(let j = 0; j < result.cols; j++) {
                    result.matrix[i][j] = m.matrix[i][j] - n;
                }
            }
        } 

        return result;
    } 

    static multiply(a, b) {
        if(!isNaN(a) && !isNaN(b)) {
            return a * b;
        }

        if(a && b instanceof Matrix) {
            if(a.cols === b.rows) {
                let result = new Matrix(a.rows, b.cols);

                for(let i = 0; i < result.rows; i++) {
                    for(let j = 0; j < result.cols; j++) {
                        let sum = 0;
                        for(let k = 0; k < a.cols; k++) {
                            sum += a.matrix[i][k] * b.matrix[k][j];
                        }
                        result.matrix[i][j] = sum;
                    }
                }

                return result;

            } else if(a.rows === b.rows && a.cols === b.cols) {
                let result = new Matrix(a.rows, a.cols);

                for(let i = 0; i < result.rows; i++) {
                    for(let j = 0; j < result.cols; j++) {
                        result.matrix[i][j] = a.matrix[i][j] * b.matrix[i][j];
                    }
                }

                return result;
            } else {
                throw 'These matrices cannot be multiplied';
            }

        } else {
            let m = a;
            let n = b;

            if(!isNaN(a)) {
                m = b;
                n = a;
            }

            let result = new Matrix(m.rows, m.cols);

            for(let i = 0; i < m.rows; i++) {
                for(let j = 0; j < m.cols; j++) {
                    result.matrix[i][j] = m.matrix[i][j] * n;
                }
            }

            return result;
        }
    }

    static from_arr(arr, cols) {
        if(!cols) {
            cols = 1;
        }
        let rows = Math.ceil(arr.length / cols);

        let result = new Matrix(rows, cols);

        for(let i = 0; i < rows; i++) {
            for(let j = 0; j < cols; j++) {
                let index = i * cols + j;
                if(isNaN(arr[index])) {
                    result.matrix[i][j] = 0;
                } else {
                    result.matrix[i][j] = arr[index];
                }
            }
        }

        return result;
    }

    static to_arr(m) {
        let arr = [];

        for(let i = 0; i < m.rows; i++) {
            for(let j = 0; j < m.cols; j++) {
                arr.push(m.matrix[i][j]);
            }
        }

        return arr;
    }

    static map(m, f) {
        let result = new Matrix(m.rows, m.cols);
        result.map((e, i, j) => f(m.matrix[i][j]));

        return result;
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {
        if(typeof data === 'string') {
            data = JSON.parse(data);
        }

        let m = new Matrix(data.rows, data.cols);
        m.matrix = data.matrix;

        return m;
    }
}