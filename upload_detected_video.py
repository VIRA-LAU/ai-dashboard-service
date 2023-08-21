import boto3

s3 = boto3.client('s3')
s3.upload_file('datasets/concatenated/04183.mp4', 'fitchain-ai-videos', 'detection_output/concatenated/concat_1.mp4')
