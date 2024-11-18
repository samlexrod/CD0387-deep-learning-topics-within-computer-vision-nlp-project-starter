aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
cd ./image
docker build -q -t udacity-sagemaker-hpo .
docker images
