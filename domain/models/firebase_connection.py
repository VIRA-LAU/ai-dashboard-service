import firebase_admin
from firebase_admin import credentials, storage
from persistence.repositories import paths

# Firebase initialization
cred = credentials.Certificate(paths.keys_path)
# firebase_admin.initialize_app(cred, {'storageBucket': 'fyp-interface.appspot.com'})
firebase_admin.initialize_app(cred, {"apiKey": "AIzaSyCDggt-rsnWQtpCCviuq2dgy4N1Jdh4CRI",
                                     "authDomain": "fyp-interface.firebaseapp.com",
                                     "databaseURL": "https://fyp-interface-default-rtdb.europe-west1.firebasedatabase.app",
                                     "projectId": "fyp-interface",
                                     "storageBucket": "fyp-interface.appspot.com",
                                     "messagingSenderId": "30164132086",
                                     "appId": "1:30164132086:web:04b00815444317def34cd7"})
bucket = storage.bucket()
