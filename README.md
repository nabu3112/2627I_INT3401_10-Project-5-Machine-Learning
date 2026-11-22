# Project 5 – Machine Learning

Repository for **Project 5: Machine Learning** (INT3401 / CS188-style project). The project uses Python and PyTorch to build and evaluate machine-learning models.

## Project overview

The project scaffold includes several machine-learning tasks, such as:

- **Q1 – Perceptron:** binary classification using a weight vector and the Perceptron learning rule.
- **Q2 – Regression:** learn to approximate a function from numerical input.
- **Q3 – Digit classification:** classify handwritten digits (0–9) from 28×28 grayscale images.
- **Q4 – Language identification:** classify a word by its language using a recurrent model.
- **Q5 – Convolutional digit classification:** implement a 2D convolution operation and use it in a neural network to classify handwritten digits.
- **Q6–Q7:** attention and language-model components provided by the project scaffold.

### Implemented in this submission

- **Q1:** `PerceptronModel` and its training routine.
- **Q5:** manual 2D convolution, `DigitConvolutionalModel`, its loss function, and training routine.

The status of other questions depends on the work completed by the project team; this README does not claim that they are complete.

## Main files

| File / directory | Purpose |
|---|---|
| `models.py` | Model definitions, including Perceptron and convolutional digit classifier. |
| `train.py` | Training routines for the project models. |
| `losses.py` | Loss functions used during training. |
| `backend.py` | Dataset loading and project support code. |
| `autograder.py` | Runs the project’s question-specific tests. |
| `data/` | Project data files, when included in the checkout. |
| `chargpt.py`, `gpt_model.py` | Language-model / GPT-related project components. |

## Requirements

- Windows, macOS, or Linux
- Python 3.11 (the project was set up and tested with Python 3.11)
- PyTorch
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- Jupyter and ipykernel (useful for the development environment)

## Setup on Windows with Miniconda

Open PowerShell or the VS Code terminal in the project directory.

### 1. Create and activate the environment

If the `cs188` environment already exists, activate it:

```powershell
conda activate cs188
```

If PowerShell says that `conda` is not recognized, initialise the Conda hook for the current terminal session (adjust the path if Miniconda is installed elsewhere):

```powershell
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate cs188
```

For a new environment, create it and install the dependencies:

```powershell
conda create -n cs188 python=3.11 -y
conda activate cs188
pip install torch numpy matplotlib scipy scikit-learn jupyter ipykernel
```

> PyTorch installation can differ by operating system and hardware. If the `pip install torch` command is unsuitable for your computer, use the installation command recommended by the official PyTorch installation selector.

### 2. Check Python and PyTorch

```powershell
python --version
python -c "import torch; print(torch.__version__)"
```

Make sure VS Code is using the `cs188` interpreter: **Ctrl+Shift+P → Python: Select Interpreter → select the `cs188` environment**.

## Get the project

Clone the group repository:

```bash
git clone https://github.com/nabu3112/2627I_INT3401_10-Project-5-Machine-Learning.git
cd 2627I_INT3401_10-Project-5-Machine-Learning
```

To work on the Q5 branch, if it exists on the remote:

```bash
git switch Q5
git pull origin Q5
```

If you are working on another task, switch to that task’s branch instead.

## Run the autograder

Run commands from the repository root — the directory containing `autograder.py`.

With the `cs188` environment activated:

```powershell
python autograder.py -q q1
python autograder.py -q q5
```

To run the full autograder:

```powershell
python autograder.py
```

If Conda activation is unavailable, you can call the environment’s Python executable directly (default Miniconda path on Windows):

```powershell
& "$env:USERPROFILE\miniconda3\envs\cs188\python.exe" autograder.py -q q5
```

A passing result for Q5 in this project requires the test accuracy threshold reported by the autograder (at least 80% for the supplied Q5 check). Exact accuracy can vary between runs because model initialization and batch shuffling may be random.

## Q1 and Q5 implementation notes

### Q1 – Perceptron

The Perceptron predicts one of two labels (`+1` or `-1`). It computes a score from the input features and its weight vector. During training, when a prediction is incorrect, the weight vector is updated using the Perceptron update rule:

```text
w_new = w_old + y * x
```

### Q5 – Convolutional digit classifier

The Q5 model applies a manually implemented 2D convolution to each 28×28 image using a 3×3 filter. With no padding and stride 1, the output of the convolution is 26×26. The result is flattened and passed to fully connected layers to produce 10 output scores, one for each digit class.

The training routine uses mini-batches, a Cross Entropy loss, backpropagation, and the Adam optimizer.

## Git workflow

In VS Code, the usual workflow is:

1. Check that the correct task branch is selected (for example, `Q5`).
2. Review the modified files in **Source Control**.
3. Stage the files you intend to include in the commit.
4. Enter a descriptive commit message and commit.
5. Push the branch to GitHub.

Useful Git commands:

```bash
git status
git pull origin main
git push origin Q5
```

Only merge or pull from `main` when you intend to bring the latest shared changes into your task branch. Resolve any merge conflicts before committing or pushing.
