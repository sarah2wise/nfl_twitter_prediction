#%% Startup
import tweepy
import config
import teams_nfl as Teams
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

client = tweepy.Client(
    bearer_token = config.Bearer_Token,
    consumer_key = config.API_Key,
    consumer_secret = config.API_Key_Secret,
    wait_on_rate_limit=True
    )

hoursAfterGame = 3
minsAfterGame = 12
collectingWindow = 2

game_times = pd.read_excel("GameTimes2022.xlsx")
gt = game_times[0:271] #256 for 2020, 272 for 2021, 271 for 2022
gt["Hours"] = ""
gt["Minutes"]= ""
gt['ET'] = ""
gt['startUTC'] = ""
gt['endUTC'] = ""
for ind in gt.index:  
        if gt['Time'][ind][-2:] == 'PM' and int(gt['Time'][ind].split(':')[0]) != 12:
            hours = int(gt['Time'][ind].split(':')[0]) + 12
        else:
            hours = int(gt['Time'][ind].split(':')[0])
        minutes = int(gt['Time'][ind].split(':')[1][:-2])
        gt['Hours'][ind] = hours
        gt['Minutes'][ind] = minutes
        gt['ET'][ind] = datetime(year = gt['Date'][ind].year,
                             month = gt['Date'][ind].month,
                             day = gt['Date'][ind].day,
                             hour = gt['Hours'][ind],
                             minute = gt['Minutes'][ind])
        diff = timedelta(hours=+(5+hoursAfterGame), minutes=+ minsAfterGame)
        gt['startUTC'][ind] = gt['ET'][ind]+diff
        window = timedelta(hours =+ collectingWindow)
        gt['endUTC'][ind] = gt['startUTC'][ind] + window

gt[['Time', 'Hours','Minutes', 'ET', 'startUTC', 'endUTC']].value_counts()
gt

#%% Change Team

team = 'ARI'
team_full = 'Washington Commanders'
team_chart = gt[(gt['Winner/tie'] == team_full) | (gt['Loser/tie'] == team_full)][['Week','startUTC','endUTC']].reset_index()
team_chart

df_team = pd.DataFrame()

# get team hashtags
hashtags = Teams.getTeamHashtags(team)

query = "(" + hashtags + ') -is:retweet lang:en'
print(query)

#%% Gather One Season of Data Per Team

for row in range(0,16): # 15 for BUF and CIN 2022, because of cancelled game
    start_day = str(team_chart['startUTC'][row])[0:10]
    starting_time = str(team_chart['startUTC'][row])[11:19]
    end_day = str(team_chart['endUTC'][row])[0:10]
    ending_time = str(team_chart['endUTC'][row])[11:19]
    
    start_time = start_day + 'T' + starting_time + 'Z'
    end_time = end_day + 'T' + ending_time + 'Z'
    
    print("Collecting data for " + team + " from " + start_time + ' to ' + end_time + ' for Week ' + str(team_chart['Week'][row]))
    
    tweets = "NA"
    
    while tweets == "NA":
        tweets = tweepy.Paginator(client.search_all_tweets,
                                            limit = 1,
                                            query=query, 
                                            tweet_fields=['id', 'text', 'created_at', 'public_metrics'],
                                            start_time=start_time,
                                            end_time=end_time,
                                            max_results = 100)
            
    try:
        for tweet in tweets:
            next_token = tweet.meta['next_token']
    except:
        next_token = "None"  
          
    tweetsf = tweets.flatten()
    tweet_data = []
    for tweet in tweetsf:
        data = [tweet.id, tweet.text, tweet.created_at, tweet.public_metrics['retweet_count'], tweet.public_metrics['like_count']]
        tweet_data.append(data)
    df = pd.DataFrame(data=tweet_data, columns=["id", "text", "timestamp", "retweets", "likes"]) 
    df['Week'] = team_chart['Week'][row]
    print("Number of tweets:", len(df))
    df_team = pd.concat([df,df_team])
    time.sleep(5)
    
    tweets = "NA"
    
    while next_token != 'None':
        while tweets == "NA":
            tweets = tweepy.Paginator(client.search_all_tweets,
                                            limit = 1,
                                            pagination_token = next_token,
                                            query=query, 
                                            tweet_fields=['id', 'text', 'created_at', 'public_metrics'],
                                            start_time=start_time,
                                            end_time=end_time,
                                            max_results = 100)
            
        print('New Page')
        try:
            for tweet in tweets:
                next_token = tweet.meta['next_token']
        except:
            next_token = "None" 
                
        tweetsf = tweets.flatten()
        tweet_data = []
        for tweet in tweetsf:
            data = [tweet.id, tweet.text, tweet.created_at, tweet.public_metrics['retweet_count'], tweet.public_metrics['like_count']]
            tweet_data.append(data)
        df = pd.DataFrame(data=tweet_data, columns=["id", "text", "timestamp", "retweets", "likes"]) 
        df['Week'] = team_chart['Week'][row]
        print("Number of tweets:", len(df))
        df_team = pd.concat([df,df_team])
        tweets = "NA"
        
    time.sleep(5)

df_team

#%% Save all data to file

filename = team + '_2022' + '.csv'
df_team.to_csv(filename)