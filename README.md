# aind-registration-evaluation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

This package evaluates a transformation applied in two large-scale images. After taking both images to the same coordinate system, we sample points in the intersection area that we later use to compute the metric.
The input vales are:
- Image 1: Path where image 1 is located. It could be 2D or 3D.
- Image 2: Path where image 2 is located. It could be 2D or 3D.
- Transform matrix: List of list that have the transformation matrix.
- Data type: Scale of the data. Small refers to data that can fit in memory and large for data that can't.
- Metric: Acronym of the metric that will be used for evaluation.
- Window size: For every point, we take an are equal to 2 * window_size + 1 which creates a square or cube for the two images in the same location of the intersection area.
- Sampling info: A dictionary that contains the number of points that will be sampled in the intersection area as well as the sampling type. At this moment, we sample points randomly or in a grid.

## Data type
Due to the large-scale data the Allen Institute for Neural Dynamics is generating, we are lazily reading image data and computing the metrics in chunks when the data is large (this means the data can't fit in standard computers memory). In these cases, we recommend setting the data type to `large` and `small` for images that could fit in memory. Selecting this depends on your use case, resources and nature of your data.

## Metrics
We have the most common computer vision metrics to evaluate images. Here, we include the following metrics:
| Metric         | Acronym     | Data scale |
|--------------------|---------|------------|
| Mean Squared Error | ssd     | :white_check_mark: Large :white_check_mark: Small     |
| Structural Similarity Index | ssim     | :white_check_mark: Large :white_check_mark: Small     |
| Mean Absolute Error | mae     | :white_check_mark: Large :white_check_mark: Small     |
| R2 Score | r2     | :white_check_mark: Large :white_check_mark: Small     |
| Max Error | max_err     | :white_check_mark: Large :white_check_mark: Small     |
| Normalized Cross Correlation | ncc     | :white_check_mark: Large :white_check_mark: Small     |
| Mutual Information | mi     | :white_check_mark: Large :white_check_mark: Small     |
| Normalized Mutual Information | nmi     | :white_check_mark: Large :white_check_mark: Small     |
| Information Theoretic Similarity | issm     | :white_check_mark: Small     |
| Peak Signal to Noise Ratio | psnr     | :white_check_mark: Large :white_check_mark: Small     |
| Feature Similarity Index Metric  | fsim     | :white_check_mark: Small     |

## Transform matrix
The matrix is in homogeneous coordinates. Therefore, for a 2D image the matrix will be:

$$\begin{bmatrix}
{y_{11}}&{y_{12}}&{\cdots}&{y_{1n}}\\
{x_{21}}&{x_{22}}&{\cdots}&{x_{2n}}\\
{w_{m1}}&{w_{m2}}&{\cdots}&{w_{mn}}\\
\end{bmatrix}$$ 

and for 3D:

$$\begin{bmatrix}
{z_{21}}&{x_{22}}&{\cdots}&{z_{2n}}\\
{y_{11}}&{y_{12}}&{\cdots}&{y_{1n}}\\
{x_{21}}&{x_{22}}&{\cdots}&{x_{2n}}\\
{w_{m1}}&{w_{m2}}&{\cdots}&{w_{mn}}\\
\end{bmatrix}$$

## Window size
This refers to how big the area around each sampled point will be. For example, in a 2D image the window size area for a given point will be:
![PointWindowSize](https://raw.githubusercontent.com/AllenNeuralDynamics/aind-registration_evaluation/main/images/point_window_size.png)

## Installation
To use the software, in the root directory, run
```
pip install -e .
```

To develop the code, run
```
pip install -e .[dev]
```

## To Run
Run with the following:

```
python src/eval_reg/evaluate_stitching.py
```

The example_input dict at the top of the file gives an example of the inputs. If "datatype" is set to dummy, then it will create dummy data and run it on that. If it's set to "large" then it will try to read the zarr files. The code for manipulating the zarr files is still not working.

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```
black .
```

- Use **isort** to automatically sort import statements:
```
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect the build system or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests
