Assignment 4 – Density Estimation using GAN (Deep Learning)

This assignment demonstrates how to estimate the probability density function (PDF) of a transformed dataset using a Generative Adversarial Network (GAN) implemented in PyTorch.

The project involves data preprocessing, GAN training, synthetic data generation, and density estimation using Kernel Density Estimation (KDE).

🚀 Project Overview

The goal of this assignment is to:

Load and preprocess real-world data (no2 column from dataset)

Apply a nonlinear transformation to the data

Normalize the transformed data

Train a GAN (Generator + Discriminator)

Generate synthetic samples

Compare real vs generated distributions

Estimate the final probability density using KDE

🛠️ Technologies & Libraries Used

Python

NumPy – Numerical computations

Pandas – Data handling

Matplotlib – Visualization

Scikit-learn – Kernel Density Estimation

PyTorch – GAN implementation (Deep Learning)

📂 Project Workflow
1️⃣ Data Loading
df = pd.read_csv("data.csv", encoding='latin1')
x = df["no2"].dropna().values.astype(np.float32)


Extracts no2 column

Removes missing values

Converts to float format

2️⃣ Nonlinear Transformation

The data is transformed using:

𝑧
=
𝑥
+
𝑎
𝑟
⋅
sin
⁡
(
𝑏
𝑟
⋅
𝑥
)
z=x+a
r
	​

⋅sin(b
r
	​

⋅x)

Where:

a_r and b_r are computed using roll number

This creates a modified nonlinear dataset

3️⃣ Data Normalization
z_norm = (z - z_mean) / z_std


Normalization ensures stable GAN training.

🤖 GAN Architecture
Generator

Input: Random noise (1D)

Architecture:

Linear(1 → 32)

ReLU

Linear(32 → 32)

ReLU

Linear(32 → 1)

Discriminator

Input: Real or Fake data

Architecture:

Linear(1 → 32)

LeakyReLU

Linear(32 → 32)

LeakyReLU

Linear(32 → 1)

Sigmoid

⚙️ Training Details

Loss Function: Binary Cross Entropy (BCELoss)

Optimizer: Adam

Learning Rate: 0.0002

Epochs: 4000

Batch Size: 128

Device: GPU (if available) else CPU

Training process:

Train Discriminator on real & fake data

Train Generator to fool Discriminator

Repeat for multiple epochs

📈 Results & Visualization
1️⃣ Histogram Comparison

Real transformed data distribution

GAN generated data distribution

Visual comparison of PDFs

plt.hist(z, bins=80, density=True)
plt.hist(gen_z, bins=80, density=True)

2️⃣ Kernel Density Estimation (KDE)

After generating 10,000 samples:

kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(gen_z)


Estimated final probability density

Smooth PDF curve plotted

📊 Final Output

✔️ GAN successfully learns the transformed data distribution
✔️ Generated samples closely match real distribution
✔️ KDE provides smooth density estimation

🎯 Learning Outcomes

Understanding GAN architecture

Implementing Generator & Discriminator in PyTorch

Working with adversarial training

Performing density estimation using KDE

Comparing real vs synthetic distributions

👨‍💻 Author

Priyam Chaudhary
B.E. Computer Science & Engineering
