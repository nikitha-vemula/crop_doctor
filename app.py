from flask import Flask,Blueprint , render_template , url_for , redirect , request , session
import bcrypt
import random
import smtplib
import pymongo
# from flask_pymongo import PyMongo


def create_app():
    app=Flask(__name__)
    
    from views import views
    app.register_blueprint(views,url_prefix="/")

    from auth import auth
    app.register_blueprint(auth,url_prefix="/")


    return app

if __name__=="__main__":
    app=create_app()
    app.run(debug=True)

    