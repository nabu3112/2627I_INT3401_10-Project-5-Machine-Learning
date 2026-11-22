# Project 5 – Machine Learning

This repository contains our group's implementation of **Project 5: Machine Learning** for **INT3401 / CS188-style coursework**. The project uses Python and PyTorch to implement and evaluate several machine-learning models.

## Project scope

Our submission covers **Questions 1–6**. **Question 7 (Character GPT) is outside the scope of this submission and is not required for our group's deliverable.**

| Question | Topic | Summary |
|---|---|---|
| **Q1** | Perceptron | Binary classification with labels `+1` and `-1`, using the Perceptron learning rule. |
| **Q2** | Regression | A neural network that learns to approximate a real-valued function. |
| **Q3** | Digit classification | Classification of 28×28 grayscale handwritten-digit images into classes 0–9. |
| **Q4** | Language identification | Classification of a word's language from its character sequence using a recurrent model. |
| **Q5** | Convolutional digit classification | Manual 2D convolution followed by a neural network to classify handwritten digits. |
| **Q6** | Attention | Implementation of causal self-attention with query, key, and value projections. |
| **Q7** | Character GPT | Not included in this submission. |

## Repository structure

| File / directory | Purpose |
|---|---|
| `models.py` | Model definitions for the project questions. |
| `train.py` | Training routines for the models. |
| `losses.py` | Loss functions used by the training routines. |
| `backend.py` | Dataset loading and supporting project code. |
| `autograder.py` | Runs the question-specific checks. |
| `data/` | Project datasets, if included in the checkout. |

Other support files may be present in the repository and are used by the project infrastructure.

## Requirements

- Python 3.11 (or the Python version specified by the course/project environment)
- PyTorch
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- Jupyter and ipykernel (useful for the development environment)

## Setup

The following instructions use **Miniconda on Windows**. On macOS or Linux, activate the Conda environment using the equivalent shell command.

### 1. Create and activate the environment

If the `cs188` environment already exists:

```powershell
conda activate cs188
```

If PowerShell does not recognize `conda`, initialise Conda for the current terminal session (adjust the path if Miniconda is installed elsewhere):

```powershell
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate cs188
```

To create a new environment instead:

```powershell
conda create -n cs188 python=3.11 -y
conda activate cs188
pip install torch numpy matplotlib scipy scikit-learn jupyter ipykernel
```

PyTorch installation can vary by operating system and hardware. If needed, use the installation command recommended by the official PyTorch installation selector.

### 2. Verify the environment

```powershell
python --version
python -c "import torch; print(torch.__version__)"
```

In VS Code, select the same environment with **Ctrl+Shift+P → Python: Select Interpreter → `cs188`**.

## Running the project

Run commands from the repository root — the directory containing `autograder.py`.

With the `cs188` environment activated, run the questions included in this submission individually:

```powershell
python autograder.py -q q1
python autograder.py -q q2
python autograder.py -q q3
python autograder.py -q q4
python autograder.py -q q5
python autograder.py -q q6
```

Question 7 is intentionally omitted because it is not part of this submission.

If Conda activation is unavailable in the VS Code terminal, use the environment's Python executable directly (default Miniconda path on Windows):

```powershell
& "$env:USERPROFILE\miniconda3\envs\cs188\python.exe" autograder.py -q q1
```

Replace `q1` with `q2`, `q3`, `q4`, `q5`, or `q6` to check another question.

For Q5, the supplied autograder check requires at least **80% test accuracy** for full credit. Training results can vary between runs because model initialization and data shuffling may be random.

## Implementation notes

- **Q1 – Perceptron:** computes a score from input features and a weight vector, predicts `+1` or `-1`, and updates the weights when a training example is misclassified.
- **Q2 – Regression:** trains a neural network to approximate a real-valued target function.
- **Q3 – Digit classification:** produces scores for the ten handwritten-digit classes.
- **Q4 – Language identification:** processes a sequence of characters and predicts one of the supported languages.
- **Q5 – Convolutional digit classification:** applies a manually implemented 2D convolution to each 28×28 image using a 3×3 filter. With no padding and stride 1, the convolution output is 26×26; it is then flattened and passed through fully connected layers to produce ten class scores. The training routine uses a loss function, backpropagation, and an optimizer.
- **Q6 – Attention:** computes causal self-attention using query, key, and value projections and a causal mask.

## Git workflow

In VS Code:

1. Open the local repository folder.
2. Check that the intended branch is selected.
3. Review the changed files in **Source Control**.
4. Stage the files you intend to include.
5. Enter a descriptive commit message and select **Commit**.
6. Select **Push** (or **Publish Branch** if the branch has not yet been published).

Useful commands:

```bash
git status
git pull origin main
git push origin <branch-name>
```

Pull from `main` only when you intend to bring the latest shared changes into your working branch. Resolve any merge conflicts before committing or pushing.
