# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.5.1-cpu-py36-ubuntu16.04
RUN pip install smdebug

# Set the environment variable to include your code in the PATH
ENV PATH="/opt/ml/code:${PATH}"

# This environment variable is used by the SageMaker PyTorch container
# to determine our user code directory
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Copy the user code to the container
COPY hpo-container.py /opt/ml/code/hpo-container.py

# Set the working directory
WORKDIR /opt/ml/code

# Define the entry point for the container
ENV SAGEMAKER_PROGRAM hpo-container.py
