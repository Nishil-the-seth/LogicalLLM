# Week 1 – MLP from Scratch

This is a simple neural network (MLP) implemented in pure Python without NumPy. It is trained on the two moons dataset using mini-batch SGD.

## Files

- `mlp.py` – MLP class with forward and backpropagation
- `dataset.py` – Generates the two moons dataset
- `train.py` – Trains the model and plots loss + decision boundary
- `plot.py` – Functions to plot loss curve and decision boundary

## How to Run

```bash
pip install matplotlib scikit-learn
python train.py
```

## Output

- A **loss curve** showing how the model learns
- A **decision boundary plot** showing how the model separates classes

---

This was done as part of Week 1 of the MLP-from-scratch project.
