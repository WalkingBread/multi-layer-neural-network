# multi-layer-neural-network

This is a multi-layer neural network library created in pure JavaScript. (still in development)

## Usage:

### Layer creation:
```js
const layer = NeuralNetwork.createLayer({
    nodes: 2,
    activation: 'sigmoid'
});
```

### Neural Network initialization:
```js
const brain = NeuralNetwork.createNeuralNetwork({
    inputNodes: 2,
    layers: [
        NeuralNetwork.createLayer({
            nodes: 2,
            activation: 'sigmoid'
        }),
        NeuralNetwork.createLayer({ // output layer
            nodes: 1,
            activation: 'sigmoid'
        })
    ],
    learningRate: 0.1,
});
```

### Training:
```js
const dataset = [
    {
        inputs: [0, 1],
        targets: [1]
    },
    {
        inputs: [1, 0],
        targets: [1]
    },
    {
        inputs: [1, 1],
        targets: [0]
    },
    {
        inputs: [0, 0],
        targets: [0]
    }
];

for(let i = 0; i < 10000; i++) { // how many times we want it to be trained
    for(let data of dataset) {
        brain.train(data.inputs, data.targets);
    }
}
```

### Predicting:
```js
const output = brain.predict([0, 1]);
console.log(output);
```
