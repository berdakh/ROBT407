# ROBT407: Deep Learning Fundamentals with PyTorch

This repository contains educational materials for the ROBT407 course, focusing on the fundamentals of deep learning using PyTorch. It includes Jupyter notebooks covering key topics such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and practical assignments.

## Repository Contents

* **CNN-Introduction.ipynb**: An introductory notebook on Convolutional Neural Networks, demonstrating their architecture and application using the MNIST dataset.
* **Simple\_RNN.ipynb**: A notebook exploring Recurrent Neural Networks, illustrating how they process sequential data.
* **hw\.ipynb**: A homework assignment notebook designed to reinforce concepts covered in the course.
* **mnist.npz**: A compressed NumPy file containing the MNIST dataset used in the CNN notebook.
* **images/**: A directory containing images utilized within the notebooks for illustrative purposes.

## Getting Started

To begin working with the materials:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/berdakh/ROBT407.git
   cd ROBT407
   ```

2. **Set Up the Environment**:

   Ensure you have Python 3.x installed. It's recommended to use a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Required Packages**:

   Install the necessary Python packages using pip:

   ```bash
   pip install torch torchvision matplotlib numpy jupyter
   ```

4. **Launch Jupyter Notebook**:

   Start the Jupyter Notebook server to access the notebooks:

   ```bash
   jupyter notebook
   ```

   Navigate to the desired notebook (e.g., `CNN-Introduction.ipynb`) to begin exploring the content.

## Prerequisites

* **Python 3.x**: Ensure Python is installed on your system.
* **PyTorch**: Deep learning framework used for building and training models.
* **Jupyter Notebook**: An interactive environment for running and editing notebooks.

