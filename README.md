**[NEW!] 2022 Ego4D Challenges now open for Forecasting**
- [Short Term Anticipation](https://eval.ai/web/challenges/challenge-page/1623/overview) (deadline June 1 2022)
- [Long Term Anticipation](https://eval.ai/web/challenges/challenge-page/1598/overview) (deadline June 1 2022)
- [Future Hand Prediction](https://eval.ai/web/challenges/challenge-page/1630/overview) (deadline Oct 1 2022)


# EGO4D Forecasting Benchmark

This repository contains code to replicate the results of the [EGO4D Forecasting Benchmark](https://ego4d-data.org/docs/benchmarks/forecasting/) in [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058).

[EGO4D](https://ego4d-data.org/docs/) is the world's largest egocentric (first person) video ML dataset and benchmark suite.

For more information on Ego4D or to download the dataset, read: [Start Here](https://ego4d-data.org/docs/start-here/).

## Installation
This code requires Python>=3.7 (this a requirement of pytorch video). If you are using Anaconda, you can create a clean virtual environment with the required Python version with the following command:

`conda create -n ego4d_forecasting python=3.7`

To proceed with the installation, you should then activate the virtual environment with the following command:

`conda activate ego4d_forecasting`

Run the following commands to install the requirements:

`cat requirements.txt | xargs -n 1 -L 1 pip install`

In order to make the `ego4d` module loadable, you should add the current directory to the Python path:

`export PYTHONPATH=$PWD:$PYTHONPATH`

Please note that the command above is not persistent and hence you should run it every time you open a new shell.

## Using the code
Please refer to the following README files for the benchmark specific code/instructions:
 * [Short-Term Object Interaction Anticipation](SHORT_TERM_ANTICIPATION.md)
 * [Long-Term Action Anticipation](LONG_TERM_ANTICIPATION.md)
 * [Future Hand Prediction](Ego4D-Future-Hand-Prediction/README.md)
