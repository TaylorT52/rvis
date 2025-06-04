#!/bin/bash

# Create mnist directory if it doesn't exist
mkdir -p mnist

# Download MNIST files from official mirror
echo "Downloading MNIST dataset..."
curl -L -o mnist/train-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
curl -L -o mnist/train-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
curl -L -o mnist/t10k-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
curl -L -o mnist/t10k-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

# Extract the files
echo "Extracting files..."
gunzip -f mnist/*.gz

echo "Done! MNIST dataset is ready in the mnist directory." 