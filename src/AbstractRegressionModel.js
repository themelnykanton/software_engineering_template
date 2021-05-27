import * as tf from '@tensorflow/tfjs';

export default class AbstractRegressionModel {

    constructor(
        width,
        height,
        optimizerFunction = tf.train.sgd,
        maxEpochPerTrainSession = 100,
        learningRate = 0.1,
        expectedLoss = 0.001
    ) {
        this.width = width;
        this.height = height;
        this.optimizerFunction = optimizerFunction;
        this.expectedLoss = expectedLoss;
        this.learningRate = learningRate;
        this.maxEpochPerTrainSession = maxEpochPerTrainSession;

        this.initModelVariables();

        this.trainSession = 0;
        this.epochNumber = 0;
        this.history = [];
    }

    initModelVariables() {
        throw Error('Model variables should be defined')
    }

    f() {
        throw Error('Model should be defined')
    }

    xNormalization = xs => xs.map(x => x / this.width);
    yNormalization = ys => ys.map(y => y / this.height);
    yDenormalization = ys => ys.map(y => y * this.height);

    /**
     * Calculate loss function as mean-squared deviation
     *
     * @param predictedValue - tensor1d - predicted values of calculated model
     * @param realValue - tensor1d - real value of experimental points
     */
    loss = (predictedValue, realValue) => {
        const loss = predictedValue.sub(realValue).square().mean();
        this.lossVal = loss.dataSync()[0];
        return loss;
    };

    /**
     * Train model until explicitly stop process via invocation of stop method
     * or loss achieve necessary accuracy, or train achieve max epoch value
     *
     * @param x - array of x coordinates
     * @param y - array of y coordinates
     * @param callback - optional, invoked after each training step
     */
    async train(x, y, callback) {
        const currentTrainSession = ++this.trainSession;
        this.lossVal = Number.POSITIVE_INFINITY;
        this.epochNumber = 0;
        this.history = [];

        // convert data into tensors
        const input = tf.tensor1d(this.xNormalization(x));
        const output = tf.tensor1d(this.yNormalization(y));

        while (
            currentTrainSession === this.trainSession
            && this.lossVal > this.expectedLoss
            && this.epochNumber <= this.maxEpochPerTrainSession
            ) {
            const optimizer = this.optimizerFunction(this.learningRate);
            optimizer.minimize(() => this.loss(this.f(input), output));
            this.history = [...this.history, {
                epoch: this.epochNumber,
                loss: this.lossVal
            }];
            callback && callback();
            this.epochNumber++;
            await tf.nextFrame();
        }
    }

    stop() {
        this.trainSession++;
    }

    /**
     * Predict value basing on trained model
     *  @param x - array of x coordinates
     *  @return Array({x: integer, y: integer}) - predicted values associated with input
     *
     * */
    predict(x) {
        const input = tf.tensor1d(this.xNormalization(x));
        const output = this.yDenormalization(this.f(input).arraySync());
        return output.map((y, i) => ({ x: x[i], y }));
    }
}
