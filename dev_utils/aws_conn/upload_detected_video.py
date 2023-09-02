import boto3
import os
import persistence.repositories.paths as path


def upload_highlights_to_s3(game_id: str):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket='fitchain-ai-videos',
                                  Prefix=f'detection_output/highlights/{game_id}/')
    files_in_s3 = []
    if 'Contents' in response:
        files_in_s3 = [os.path.basename(obj['Key']) for obj in response['Contents']]
    for vid in os.listdir(path.highlights_path / game_id):
        if vid not in files_in_s3:
            s3.upload_file(path.highlights_path / game_id / vid, 'fitchain-ai-videos',
                           f'detection_output/highlights/{game_id}/{vid}')
            s3.put_object_acl(ACL='public-read', Bucket='fitchain-ai-videos',
                              Key=f'detection_output/highlights/{game_id}/{vid}')


if __name__ == '__main__':
    upload_highlights_to_s3('f1tch41n')
