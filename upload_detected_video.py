import boto3

s3 = boto3.client('s3')

s3.upload_file('datasets/videos_input/04181.mp4', 'fitchain-ai-videos', 'detection_output/04181.mp4')
