# Multi-class Multi-Model Dog Breed Classification

This project is designed to train a multi-class multi-model image classification model using the dog breed classification dataset. The project uses the AWS SageMaker service to train the model and deploy it as an endpoint. The project also uses the SageMaker Debugger and Profiler to profile and debug the model during training. AWS Clarify is used to explain the model's predictions. It is designed to be a starting point for building a more complex image classification model using AWS SageMaker. Therefore accuracy is not the main goal of this project.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
- Download the starter files.
- Download/Make the dataset available. 

> The following notebook is designed to be run in SageMaker Studio. It is recommended to run the notebook in SageMaker Studio to avoid any compatibility issues: [train_and_deploy.ipynb](train_and_deploy.ipynb)

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom. The dataset contains images of 133 different dog breeds. The dataset is divided into training, validation, and test sets. The dataset is structured as follows:

```
- dogImages
  - train
    - 001.Affenpinscher
    - 002.Afghan_hound
    - ...
  - test
    - 001.Affenpinscher
    - 002.Afghan_hound
    - ...
  - valid
    - 001.Affenpinscher
    - 002.Afghan_hound
    - ...
```

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. You can download it locally and then upload it to S3. Or you can use wget to download the data directly to the SageMaker instance and use boto3 to upload it to S3. The code is included in the notebook.

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

The model used for this experiment is a pretrained ResNet18 model. The model is trained using transfer learning. This model was chosen because it is a good starting point for image classification tasks. The model is pretrained on the ImageNet dataset and can be fine-tuned on the dog breed classification dataset.


## Debugging and Profiling
During the final testing of the model, the model was profiled and debugged to ensure that it was working as expected. 

This was the final testing job run:
![Best Hyperparameter Evaluation](best-hyperparameter-eval.png)

The testing captured profiling and debugging information for the first 50 steps of the training job on a 5000ms interval capturing the following information:
- System metrics
- Framework metrics
- Detailed profiling information

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

The profiling and debugging information showed that the model was working as expected and that the training job was running smoothly.

However, there were several factors to consider:
- The CPU utilization was very low, which could indicate that the model was not complex enough to require a lot of resources. Ideally, we would want to use GPU instances for training the model.

- The initialization took a long time due to s3 data transfer to disk. Perhaps using boto3 to download the data is not the most efficient way to get the data. Using aws cli to download the data to the instance might be a better option. For example:

  
  ```
  aws s3 cp s3://my-bucket/dataset /local/data --recursive --only-show-errors --no-progress --multipart-chunk-size 8MB
  ```
  or 
  ```
  aws s3 cp --endpoint-url https://s3-accelerate.amazonaws.com s3://my-bucket/dataset /local/data
  ```

  - GPU was not set up for the training job due to cost implications. This could have sped up the training process significantly. This would optimize the heavy convolutional operations in the model.

  - There was more time allocated to evaluating the model rather than training it. This could be adjusted to allocate more time to training the model by reducing the frequency of evaluation.

See the [profiler report](ProfilerReport/profiler-output/profiler-report.html) for more information.

## Model Deployment

The first model is a ResNet18 model that is pretrained on the ImageNet dataset. The model is fine-tuned on the dog breed classification dataset. The model is first registered with the model registry and then deployed as an endpoint using the SageMaker SDK. The model is stored in an S3 bucket and the code used to deploy the model is in the notebook.

The second model is a mobilenetv2 model that is optimized for mobile devices. This model is also registered with the model registry and deployed as an endpoint using the SageMaker SDK.

 The Multi-Model code used to create the endpoint is in the notebook.

The models can be queried using the following code:

```python
import json

runtime_client = boto3.client("sagemaker-runtime")

endpoint_name = "dog-breed-classifier-multi-endpoint"
image_path = "dogImages/test/001.Affenpinscher/Affenpinscher_00003.jpg"

# Load the image as bytes
with open(image_path, "rb") as f:
    payload = f.read()

# Invoke the endpoint for core model
response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    TargetModel="dog-breed-classifier/model.tar.gz",
    ContentType="application/x-image",
    Body=payload,
)

# Decode the response
result = response["Body"].read()
print("Prediction result:", json.loads(result))

# Invoke the endpoint for mobile model
response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    TargetModel="dog-breed-classifier-mobile/model.tar.gz",
    ContentType="application/x-image",
    Body=payload,
)

# Decode the response
result = response["Body"].read()
print("Prediction result:", json.loads(result))
```


Here is a screenshot of the deployed active endpoint in Sagemaker.
![Multi Model](multi-model-endpoint.png)

Here are the screenshots of the hyperparameter tuning job that shows at least 2 different training runs and their results.

![Hyperparameter Tuning](hyperparameter-run.png)
![Hyperparameter Tuning](hyperparameter-tuning.png)
![Hyperparameter Tuning](hyperparameter-tuning-best.png)

## Standout Suggestions

**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

