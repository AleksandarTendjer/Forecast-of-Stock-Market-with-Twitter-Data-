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

from datetime import datetime
from datetime import timedelta

import time

def sleep_minutes(minutes):
    time.sleep(minutes * 60)

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
    consumerKey = "vc6hkKHFPxzbbS3KTlffDCoCU"
    consumerSecret = "uM4VwTdl4pTthJyPnPdteCiuRzWKEMxSfqxuLf6SYMmlrNeEnv"
    accessToken = "1280967822739464193-cnAPJWOW98OcnpeJc9QUAb060ALPmm"
    accessTokenSecret = "W8HcQm4cDw3d8IYJ7Iebq9dEeNFZA1eIKxiPS93kP5RQf"
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    ###########1. Data Collection###############
    #
    #keyword = input("Please enter keyword or hashtag to search: ")
    keyword =company
    #noOfTweet = int(input("Please enter how many tweets to analyze: "))
    noOfTweet = 100
   
    day_before_date = "2021-05-01"
    stop_date="2021-05-18"
    start = datetime.strptime(day_before_date, "%Y-%m-%d")
    day_after = datetime.strptime(day_before_date, "%Y-%m-%d") + timedelta(days=1)   # increase day one by one
    stop = datetime.strptime(stop_date, "%Y-%m-%d")
    i=0
    while start < stop:
        start = start + timedelta(days=1)  # increase day one by one
        day_after = start + timedelta(days=1)  # increase day one by one
        str_start=str(start.year)+'-'+str(start.month)+'-'+str(start.day)
        str_after=str(day_after.year)+'-'+str(day_after.month)+'-'+str(day_after.day)

        tweets = tweepy.Cursor(api.search, q=keyword,since=str_start,until=str_after).items(noOfTweet)
        
        i+=1
        tweets_writer=None
        with io.open('tweets_file_'+company+'_'+str(start.month)+'_'+str(start.day)+'.csv','a+', encoding="utf-8") as tweets_file:
            tweets_writer = csv.writer(tweets_file, delimiter=',', lineterminator = '\n')
            tweets_writer.writerow(["Text:"])
            print(tweets_writer)
            if tweets_writer:
                for tweet in tweets:
                    tweets_writer.writerow([tweet.text])
            tweets_file.close()
        print("done"+str(i)+"times")
        # sleep 15 minutes to download more data from twitter
        sleep_minutes(15)
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