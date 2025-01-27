
# Quantum Generative Adversarial Network (QGAN) with PyTorch and PennyLane

This repository contains a PyTorch implementation of a Quantum Generative Adversarial Network (QGAN) designed for generating handwritten digits using the Optical Recognition of Handwritten Digits dataset. The project combines classical neural networks and quantum circuits to achieve generative modeling.

## Features
- **Custom Dataset Loader**: A PyTorch dataset class to load and preprocess the handwritten digits dataset.
- **Classical Discriminator**: A fully connected feedforward neural network that classifies real and fake images.
- **Quantum Generator**: A quantum-based generator using PennyLane that employs the patch method with multiple sub-generators.
- **Training Loop**: An adversarial training loop that alternates between training the discriminator and generator.

## Requirements
- Python 3.8+
- PyTorch
- PennyLane
- NumPy
- Pandas
- matplotlib

Install the dependencies with:
```bash
pip install torch pennylane numpy pandas matplotlib
```

## Code Structure
- `DigitsDataset`: A PyTorch dataset class for loading and filtering the dataset by label.
- `Discriminator`: A fully connected neural network for binary classification.
- `PatchQuantumGenerator`: A quantum-based generator leveraging parameterized quantum circuits.
- `quantum_circuit`: Implements a quantum circuit with repeated layers of parameterized gates and controlled-Z gates.
- `partial_measure`: Non-linear post-processing of quantum circuit outputs.

## Usage

### Loading the Dataset
The `DigitsDataset` class reads a CSV file containing the digit images and filters them by label:
```python
dataset = DigitsDataset(csv_file="path/to/dataset.csv", transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
```

### Training the Model
Set hyperparameters and initialize the discriminator and generator:
```python
discriminator = Discriminator().to(device)
generator = PatchQuantumGenerator(n_generators=4).to(device)
```

Train the models using the adversarial training loop:
```python
while counter < num_iter:
    for data, _ in dataloader:
        # Training steps for discriminator and generator
        ...
```

### Visualizing Results
Generated images can be visualized by plotting the outputs from the generator:
```python
import matplotlib.pyplot as plt

for img in results[-1]:
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
```

## Quantum Circuit
The quantum generator uses a parameterized circuit with depth and ancillary qubits:
```python
@qml.qnode(dev, diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    ...
```

## License
This project is licensed under the MIT License.
