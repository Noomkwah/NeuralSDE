# NeuralSDE
A project based on the article [Robust Pricing and Hedging via Neural SDEs](https://arxiv.org/abs/2007.04154) (see the Github of the author [here](https://github.com/msabvid/robust_nsde)), that implements the main ideas developed and reproduces its results. The whole work has been carried out in Python, with PyTorch, in collaboration with [Anna Liffran](https://www.linkedin.com/in/anna-liffran-2b6a1b244/).

## Architecture of the project

The whole project principal code is gathered in the single file `neural_models.py`, which implements multiple classes of Neural Stochastic Differential Equations models (`NeuralSDE`, `NeuralLVmodel` and `NeuralLSVmodel`) as well as a `NeuralTrainer` that eases the training of such networks. All classes, methods and functions are accompanied by clear (I hope) docstrings. Plus, in case of impropre use of the declared classes, the correct Error messages shall be thrown, to help the user to get familiar with these models and quickly find any coding issue.

The file `basic_models.py` has two goals. It both helped at generating the data used for calibration of considered models, thanks to the class `HestonModel`, and draws Implied Volatility Surfaces (through a dedicated method of the `BlackScholesModel`).

The `data` folder contains three files:
1. `generate_data.ipynb`: Script that generate the data I used.
2. `Call_prices_59.pt`: The data used by the original authors.
3. `Call_Put_prices.csv`: The data I used. I followed the same methodology as the authors, so my data are really similar to the authors' one.

The `models` folder contains four kind of files:
1. `train_networks.ipynb`: Script that trains models (quite straight-forward).
2. `MODEL_problem_precision.pt`: Models already trained on the considered *problem*. Calibration to training data (i.e to `Call_Put_prices.csv`) has been done with a Mean Square Error (MSE) of *precision*.
3. `records_MODEL_problem_precision.pt`: Evolution of losses (MSE and control variate variance) during training.
4. `logs_MODEL_problem_precision.txt`: The logs printed out during training. The data in `records_MODEL_problem_precision.pt` corresponds to what is written in these logs.

The `images` folder contains:
1. `plot_results.ipynb`: Script that produces results and plots.
2. Some images that displays multiple results.





