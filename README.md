# Micrograd: Reverse-Mode Autodiff in Small Python

Micrograd is a minimal learning project for understanding how automatic differentiation and neural networks work under the hood.

It implements:

- a scalar-based reverse-mode autodiff engine (`Value`)
- a tiny neural net stack (`Neuron`, `Layer`, `MLP`)
- notebook demos for training and computation graph tracing

## Project Layout

| Path | Purpose |
| --- | --- |
| `micrograd/engine.py` | Core scalar value type and backprop logic |
| `micrograd/nn.py` | Small neural network abstractions built on `Value` |
| `demo.ipynb` | Binary classification demo on synthetic datasets |
| `trace_graph.ipynb` | Graph visualization with Graphviz |
| `test/test_engine.py` | Gradient checks against PyTorch |

## Quick Start (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`demo.ipynb` uses `scikit-learn` (imported as `sklearn` in Python).

## Graph Visualization Prerequisite

The Python package `graphviz` is not enough by itself. You also need the Graphviz system executable (`dot`) available on PATH.

```powershell
winget install --id Graphviz.Graphviz -e
where dot
dot -V
```

## Minimal Usage Example

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
d = a * b + b**3
e = (a + b) - d
g = e**2 / 2.0
g.backward()

print(g.data)
print(a.grad, b.grad)
```

## Run the Demo Notebook

```powershell
jupyter notebook demo.ipynb
```

## Run Tests

Install PyTorch first (tests compare gradients with torch), then run:

```powershell
python -m pytest
```

## Notes

- Use `pip install scikit-learn`.
- Keep the working directory at repo root when running notebooks/scripts so `micrograd` imports resolve cleanly.
