# Unsloth-aws-Deployment
Deployment of unsloth finetuned llama3 on AWS.
model.tar.gz with inference.py is on s3 bucket. 
Docker file and requirements.txt is in the my-llama-deployment folder in notebook instance along with the notebook itself.
Notebook as end-point creation logic and inference logic.

-- S3 folder has the inference.py file. Model (downloaded from hf_repo should be there as well). Then the entire folder will be uploaded to s3 as model.tar.gz

## Commands to run on Bash (terminal)

1. Authenticate Docker to ECR:
aws ecr get-login-password --region us-east-2 | \
docker login --username AWS --password-stdin 225725557140.dkr.ecr.us-east-2.amazonaws.com

2. Build the Docker Image
docker build -t llama3-unsloth .

3. Tag the Docker Image
docker tag llama3-unsloth:latest \
225725557140.dkr.ecr.us-east-2.amazonaws.com/llama3-unsloth:latest

4. Push the image to ECR
docker push 225725557140.dkr.ecr.us-east-2.amazonaws.com/llama3-unsloth:latest

5. Package model directory for s3
cd llama3-model-dir  # your local model directory
tar -czvf model.tar.gz *

6. Upload package to s3
aws s3 cp model.tar.gz s3://unsloth-llama3/llama3-model/model.tar.gz

