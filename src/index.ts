const tf = require('@tensorflow/tfjs'); 


const DATA = tf.tensor([
    [2.0, 1.0],
    [5.0, 1.0],
    [7.0, 4.0],
    [12.0, 5.0],
])
const data = tf.expandDims(tf.tensor([5.0, 7.0, 12.0, 19.0]), 1);

const HIDDEN_SIZE = 4
const model = tf.sequential()
model.add(
  tf.layers.dense({
    inputShape: [DATA.shape[1]],
    units: HIDDEN_SIZE,
    activation: "tanh",
  })
)
model.add(
  tf.layers.dense({
    units: HIDDEN_SIZE,
    activation: "tanh",
  })
)
model.add(
  tf.layers.dense({
    units: 1,
  })
)

const ALPHA = 0.001
model.compile({
  optimizer: tf.train.sgd(ALPHA),
  loss: "meanSquaredError",
})

const train = async () => {
    await model.fit(DATA, data, {
        epochs: 200,
        callbacks: {
          onEpochEnd: async (epoch: any, logs: any) => {
            if (epoch % 10 === 0) {
              console.log(`Epoch ${epoch}: error: ${logs.loss}`)
            }
          },
        },
    })
}

train(); 

console.log(model.layers[0].getWeights()[0].shape)
model.layers[0].getWeights()[0].print()

const lastDayFeatures = tf.tensor([[12.0, 5.0]])
model.predict(lastDayFeatures).print()