import firebase_admin
from firebase_admin import credentials, storage
from persistence.repositories import paths
# Firebase initialization
cred = credentials.Certificate(paths.keys_path)
firebase_admin.initialize_app(cred, {'storageBucket': 'fyp-interface.appspot.com'})
bucket = storage.bucket()