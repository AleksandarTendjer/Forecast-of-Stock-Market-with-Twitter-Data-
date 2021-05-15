from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import io

import re
import string

import csv

dowJonesCompanies=[
'Apple' 
'UnitedHealth Group',
'Home Depot',
'Goldman Sachs', 
'Microsoft',
'Visa', 
'McDonalds',
'Boeing', 
'3M',
'Johnson & Johnson',
'Caterpillar',
'Walmart',
'Procter & Gamble Company', 
'IBM',
'The Travelers Companies',
'Walt Disney',
'JP Morgan Chase & Co',
'Nike',
'American Express', 
'Chevron',
'Merck & Company',
'Raytheon',
'Intel', 
'Verizon',
'Coca-Cola',
'Cisco', 
'DOW',
'Exxon Mobil',
'Walgreens Boots',
'Pfizer']
def scrape(company):
    # Authentication
    consumerKey = "Ka4T1VIYtEd2KybcO8vfH1bQY"
    consumerSecret = "b40HTUn9U4uqfp58nsYQnw0iZLyLj6Pt1bE0drrKzcTZ42GtOp"
    accessToken = "1280967822739464193-cpMnejPoX7ZNJFbLUVRpCqE1pv5TlO"
    accessTokenSecret = "fIwsmzBMkOB1pLEYQGCglVvLMPGPHUM3zMj5yNP12TtHI"
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    ###########1. Data Collection###############
    #
    #keyword = input("Please enter keyword or hashtag to search: ")
    keyword =company
    #noOfTweet = int(input("Please enter how many tweets to analyze: "))
    noOfTweet = 500
    tweets = tweepy.Cursor(api.search, q=keyword).items(noOfTweet)

    tweets_writer=None
    with io.open('tweets_file_'+company+'.csv','a+', encoding="utf-8") as tweets_file:
        tweets_writer = csv.writer(tweets_file, delimiter=',', lineterminator = '\n')
        tweets_writer.writerow(["Text:"])
        if tweets_writer:
            for tweet in tweets:
                tweets_writer.writerow([tweet.text])
        tweets_file.close()
if __name__ == '__main__':
    if len(sys.argv)==2:
        for comp in dowJonesCompanies:
            if sys.argv[1]==comp:
                scrape(sys.argv[1])
                exit(0)
        print(" The company is not on the Dow Jones stock market list  .")
        exit(1)
    else:
        print("Incorrect number of params. Enter only a name of the company from the Dow Jones stock market .")
        exit(1)