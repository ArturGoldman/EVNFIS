# EVNFIS
This repository contains experiments exploring proposed method EVNFIS - Empirical Variance minimisation, using Normalising Flows and Importance Sampling.

Based on PyTorch 1.6.0

Brief description of files.

---
### Technical files

- **myfl.py** - **!!!WARNING!!!** This is not my script. I havent written it. It was taken from https://github.com/ikostrikov/pytorch-flows. This script implements 
Normalising Flows architectures.

- **distribution_classes.py** - Classes of distributions used in experiments. Based on PyTorch classes

- **generate_means** - Functions for mean estimation. Both regular law of large numbers and Importance Sampling modified version

- **model_building.py** - **!!!WARNING!!!** This script is based on https://github.com/ikostrikov/pytorch-flows. Assembles models.

- **optimisation.py** - Optimises model on minimising empirical variance.

---

### Demo version of each experiment

- **Banana_shape_test.ipynb** - Jupyter Notebook, containing examples of experiments over Banana-shaped density.

- **Gaussian_Grid_test.ipynb** - Jupyter Notebook, containing examples of experiments over Gaussian-Grid density.

---

### Regular Banana-shaped grid search

- **banana_test.ipynb** - Jupyter Notebook, which performs hyperparameter grid-search and outputs data to **banana_res.csv**.

- **banana_res.csv** - result of **banana_test.ipynb**

- **banana_analysis.ipynb** - Jupyter Notebook, which performs analysis of results stored **banana_res.csv**. This script analyses variance reduction.

---

### Importance sampling-like Banana-shape grid search

- **banana_is_test.ipynb** - Jupyter Notebook, which performs hyperparameter grid-search for Importance Sampling-like functions and outputs data to 
**bananaressev.csv**.

- **bananaressev.csv** - result of **banana_is_test.ipynb**

- **banana_is_analysis.ipynb** - Jupyter Notebook, which performs analysis of results stored **bananaressev.csv**. This script analyses variance reduction.

---

### Gaussian-Grid grid search

- **2_dim.ipynb** - Jupyter Notebook, which performs hyperparameter grid-search and outputs data to **2dimressev.csv.csv**.

- **2dimressev.csv.csv** - result of **2_dim.ipynb**

- **2_dim_ananlisys.ipynb** - Jupyter Notebook, which performs analysis of results stored **2dimressev.csv.csv**. This script analyses variance reduction.
