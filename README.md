# liesel-template

A template repository for data analysis projects and simulation studies with [Liesel](https://liesel-project.org) and [RLiesel](https://github.com/liesel-devs/rliesel). It contains the following structure for you to build on:

- :file_folder: `src`: An implementation of the bivariate normal distribution based on [TensorFlow Probability](https://www.tensorflow.org/probability). The distribution is parameterized for RLiesel, which can be used to configure semi-parametric regression predictors for the marginal means, standard deviations and the correlation parameter. *Replace the files in this directory with your own Python code.*
- :file_folder: `tests`: Unit tests for the bivariate normal distribution. *Add your test code here, and it will be run automatically by [pytest](https://pytest.org).*
- :file_folder: `examples`: Some examples using the bivariate normal distribution in combination with [Quarto](https://quarto.org) and [GNU Parallel](https://www.gnu.org/software/parallel). *Replace the files in this directory with your own data analysis scripts or simulation studies. You can also create other directories outside of `src` for this purpose.*
  - :file_folder: `01-quarto`: A semi-parametric distributional regression model with a bivariate normal response variable. This example illustrates how to integrate Python and R code using Liesel and RLiesel with Quarto and [Reticulate](https://rstudio.github.io/reticulate).
  - :file_folder: `02-gnu-parallel`: A small simulation study, which is implemented using GNU Parallel.
- :page_facing_up: `environment.yml`: The specification of the [Conda](https://github.com/conda-forge/miniforge) environment for our software to run in. *Change the name of the project, and its Python and R dependencies here.*
- :page_facing_up: `pyproject.toml`: The configuration of our Python package and some Python development tools. *Change the name of the Python package, its description and your name as an author here.*
- :toolbox: Some more configuration files that should be mostly self-explanatory.

## Running our template code

We recommend using [Conda](https://github.com/conda-forge/miniforge) (or even better, [Micromamba](https://mamba.readthedocs.io/en/latest/micromamba-installation.html)) for Python and R dependency management. To run our template code, follow these steps on Linux, Mac OS X or the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/windows/wsl). Note that [JAX](https://github.com/google/jax) and hence Liesel *do not work natively on Windows*:

1. Assuming you have Conda installed, create a Conda environment with the dependencies and the local Python package installed:
```
# on linux, wsl and intel macs:
conda env create -f environment.yml -p ./env

# on apple silicon macs (due to some bugs):
conda env create -f environment-osx-arm64.yml -p ./env
conda install -c conda-forge/osx-64 -p ./env pandoc=3.1.1
conda install -c conda-forge -p ./env quarto=1.3.450
```
2. Activate the Conda environment and install RLiesel:
```
conda activate ./env
Rscript -e "remotes::install_github('liesel-devs/rliesel')"
```
3. If you were able to follow the previous steps, you should be set to run our first example:
```
quarto render examples/01-quarto/example.qmd
```
4. If Quarto or Reticulate do not use the correct Conda environment automatically, try setting the `RETICULATE_PYTHON_ENV` variable:
```
RETICULATE_PYTHON_ENV="$PWD/env" quarto render examples/01-quarto/example.qmd
```

## Developing your own project

To develop your own project based on this repository, start as follows:

1. Replace the strings `liesel-template` and `liesel_template` with the name of your project in `environment.yml`, `liesel-template.Rproj`, `pyproject.toml`, `src/liesel_template` and `tests`.
2. Remove the Conda environment `env` and repeat the steps from the previous section.

These commands might come in handy as you continue to develop your project:

- `pdoc ./src`: Serves the docs with [pdoc](https://pdoc.dev).
- `pre-commit run -a`: Runs the [pre-commit](https://pre-commit.com) hooks.
- `pytest`: Runs [pytest](https://pytest.org).

## Dependency management with Conda

This repository uses Conda for Python and R dependency management. Conda installs the environment for our software to run in (think: a fancy virtual environment for Python and R). If you need a different version of Python, R, Quarto or [GNU Parallel](https://www.gnu.org/software/parallel), or any additional Python or R packages, edit `environment.yml`.

## Simulation studies using GNU Parallel

Many simulation studies in statistics are [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel). It is straightforward to accelerate them by distributing them to a number of cores or computers. In our opinion, GNU Parallel is a great tool for parallelizing simulation studies using Liesel and RLiesel for the following reasons:

- It is a shell tool, so it can run both Python and R code, as well as Quarto documents integrating both programming languages.
- By opening and closing Python for each job, it works around potential memory leaks in JAX.

**TODO:** Add a few words about parameterized Quarto documents here.
