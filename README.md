# Optical Recognition of Handwritten Digits - Quantum GAN

This repository contains a PyTorch implementation of a Quantum Generative Adversarial Network (QGAN) designed to generate handwritten digits based on the Optical Recognition of Handwritten Digits Dataset.

## Overview

The project includes the following key components:

- **DigitsDataset**: A custom PyTorch Dataset class for loading and preprocessing the handwritten digits dataset.
- **Classical Discriminator**: A fully connected neural network that serves as the discriminator in the GAN framework.
- **Quantum Generator**: A quantum-based generator that utilizes parameterized quantum circuits to generate synthetic data.
- **Training Framework**: The implementation of the training process, including the loss functions, optimizers, and training loop.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/digits-qgan.git
   cd digits-qgan
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Make sure you have [PennyLane](https://pennylane.ai/) installed for quantum simulation.

## Dataset

The dataset used in this project is the Optical Recognition of Handwritten Digits Dataset, available in CSV format. Each sample represents an 8x8 grayscale image of a handwritten digit.

## Code Structure

- `DigitsDataset`: A custom PyTorch Dataset to load and preprocess the dataset.
- `Discriminator`: A classical fully connected neural network for distinguishing between real and fake samples.
- `PatchQuantumGenerator`: A quantum-based generator implemented with PennyLane and PyTorch.
- Training loop with losses and optimizers.

## Usage

1. **Prepare the Dataset**

   Place your dataset CSV file in the project directory and specify the path when initializing the `DigitsDataset` class.

2. **Train the QGAN**

   Run the training script to train the quantum GAN:
   ```bash
   python train.py
   ```

3. **Monitor Training**

   Loss values for the generator and discriminator are displayed during training. The generated images are periodically saved for visualization.

## Code Example

Here is an example of how the dataset and models are implemented:

```python
class DigitsDataset(Dataset):
    """PyTorch Dataset for Optical Recognition of Handwritten Digits."""

    def __init__(self, csv_file, label=0, transform=None):
        self.csv_file = csv_file
        self.transform = transform
        self.df = self.filter_by_label(label)

    def filter_by_label(self, label):
        df = pd.read_csv(self.csv_file)
        return df[df.iloc[:, -1] == label]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df.iloc[idx, :-1].values.astype(np.float32).reshape(8, 8) / 16
        label = 0
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([transforms.ToTensor()])
dataset = DigitsDataset(csv_file='digits.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
```

### Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8 * 8, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
```

### Quantum Generator

```python
class PatchQuantumGenerator(nn.Module):
    def __init__(self, n_generators, q_delta=1):
        super().__init__()
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(6 * 5), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        patch_size = 2 ** (5 - 1)
        images = torch.Tensor(x.size(0), 0).to(device)

        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            images = torch.cat((images, patches), 1)

        return images
```

## Training Results

During training, the losses for the generator and discriminator are logged. The generated images are saved periodically for qualitative evaluation.

## References

- [PennyLane Documentation](https://pennylane.ai/documentation)
- [Optical Recognition of Handwritten Digits Dataset](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

