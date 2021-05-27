import React, { useState, useEffect } from 'react';

import Canvas from './components/Canvas';
import LossPlot from './components/LossPlot_v3';
import LinearRegressionModel from './model/LinearRegressionModel';

import './RegressionModel.scss';

const WIDTH = 400;
const HEIGHT = 400;
const LINE_POINT_STEP = 5;

const predictedInput = Array.from({ length: WIDTH / LINE_POINT_STEP + 1 })
    .map((v, i) => i * LINE_POINT_STEP);

const model = new LinearRegressionModel(WIDTH, HEIGHT);

export default () => {
    const [points, changePoints] = useState([]);
    const [curvePoints, changeCurvePoints] = useState([]);
    const [lossHistory, changeLossHistory] = useState([]);

    useEffect(() => {
        if (points.length > 0) {
            const input = points.map(({ x }) => x);
            const output = points.map(({ y }) => y);

            model.train(input, output, () => {
                changeCurvePoints(() => model.predict(predictedInput));
                changeLossHistory(() => model.history);
            });
        }
    }, [points]);

    return (
        <div className="regression-low-level">
            <div className="regression-low-level__top">
                <div className="regression-low-level__workarea">
                    <div className="regression-low-level__canvas">
                        <Canvas
                            width={WIDTH}
                            height={HEIGHT}
                            points={points}
                            curvePoints={curvePoints}
                            changePoints={changePoints}
                        />
                    </div>
                    <div className="regression-low-level__toolbar">
                        <button
                            className="btn btn-red"
                            onClick={() => model.stop()}>Stop
                        </button>
                        <button
                            className="btn btn-yellow"
                            onClick={() => {
                                model.stop();
                                changePoints(() => []);
                                changeCurvePoints(() => []);
                            }}>Clear
                        </button>
                    </div>
                </div>
                <div className="regression-low-level__loss">
                    <LossPlot
                              loss={lossHistory}/>
                </div>

            </div>
        </div>
    )
}
