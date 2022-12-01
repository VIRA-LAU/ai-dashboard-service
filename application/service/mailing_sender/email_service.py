import smtplib
import ssl
from email.message import EmailMessage
import os
import firebase_admin
from firebase_admin import credentials, storage, firestore
from persistence.repositories import paths
import domain.models.firebase_connection

# cred = credentials.Certificate(paths.keys_path)
# firebase_admin.initialize_app(cred, {"apiKey": "AIzaSyCDggt-rsnWQtpCCviuq2dgy4N1Jdh4CRI",
#                                      "authDomain": "fyp-interface.firebaseapp.com",
#                                      "databaseURL": "https://fyp-interface-default-rtdb.europe-west1.firebasedatabase.app",
#                                      "projectId": "fyp-interface",
#                                      "storageBucket": "fyp-interface.appspot.com",
#                                      "messagingSenderId": "30164132086",
#                                      "appId": "1:30164132086:web:04b00815444317def34cd7"})
database = firestore.client()
usersref = database.collection('users')
docs = usersref.stream()

email_sender = "maria.hannaaa01@gmail.com"
email_password = 'bwzhxlxdijvzmrls'


def send_mail(userId: str, link: str): # to be replaced by user Id that we get from Pi
    email_receiver = ""
    for doc in docs:
        if (doc.id == userId):
            email_receiver = doc.email
    subject = 'Link of your highlights'
    body = f"""
    Dear student,
    Thank you for participating in the test development of our application.
    Here is the link of a video showing the highlights of your game:
    {link}
    Hope you enjoyed the experience!
    
    with love
    """
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
