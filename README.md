# Unsloth-aws-Deployment
Deployment of unsloth finetuned llama3 on AWS.
model.tar.gz with inference.py is on s3 bucket. 
Docker file and requirements.txt is in the my-llama-deployment folder in notebook instance along with the notebook itself.
Notebook as end-point creation logic and inference logic.

-- S3 folder has the inference.py file. Model (downloaded from hf_repo should be there as well). Then the entire folder will be uploaded to s3 as model.tar.gz
