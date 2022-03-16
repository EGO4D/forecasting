# EGO4D Forecasting Benchmark

This repository contains code to replicate the results of the [EGO4D Forecasting Benchmark](https://ego4d-data.org/docs/benchmarks/forecasting/) in [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058).

[EGO4D](https://ego4d-data.org/docs/) is the world's largest egocentric (first person) video ML dataset and benchmark suite.

For more information on Ego4D or to download the dataset, read: [Start Here](https://ego4d-data.org/docs/start-here/).

## Installation
This code requires Python>=3.8. If you are using Anaconda, you can create a clean virtual environment with the required Python version with the following command:

`conda create -n ego4d_forecasting python=3.8`

To proceed with the installation, you should activate the virtual environment with the following command:

`conda activate ego4d_forecasting`

We provide two ways to install the repository: a manual installation and a package-based installation. 
### Manual installation
This installation is recommended if you want to modify the code in place and see the results immediately (without having to re-build). On the downside, you will have to add this repository to the PYTHONPATH environment variable manually.

Run the following commands to install the requirements:

`cat requirements.txt | xargs -n 1 -L 1 pip install`

In order to make the `ego4d` module loadable, you should add the current directory to the Python path:

`export PYTHONPATH=$PWD:$PYTHONPATH`

Please note that the command above is not persistent and hence you should run it every time you open a new shell.

### Package-based installation
This installation is recommended if you want import the code of this repo in a separate project. Following these instructions, you will install an "ego4d_forecasting" package which will be accessible in any python project.

To build and install the package run the command:

`pip install .`

To check if the package is installed, move to another directory and try to import a module from the package. For instance:

```
cd ..
python -c "from ego4d_forecasting.models.head_helper import ResNetRoIHead"
```
## Using the code
Please refer to the following README files to use the short-term and long-term anticipation code:
 * [Short-Term Object Interaction Anticipation](SHORT_TERM_ANTICIPATION.md)
 * [Long-Term Action Anticipation](LONG_TERM_ANTICIPATION.md)
