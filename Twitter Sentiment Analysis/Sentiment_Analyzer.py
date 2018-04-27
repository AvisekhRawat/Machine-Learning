from textblob import TextBlob
import pandas as pd
import tweepy
#Enter Twitter API credentials
Consumer_Key ='<Your Consumer_Key >'
Consumer_Secret= '<Your Consumer_Secret>' 
Access_Token ='<Your Access_Token>'
Access_Token_Secret ='<Your Access_Token_Secret>' 

#creating Twitter API
authobj=tweepy.OAuthHandler(Consumer_Key , Consumer_Secret)
authobj.set_access_token(Access_Token , Access_Token_Secret)
api =tweepy.API(authobj)

item=input('Enter Word: ')

tweets=api.search(item)
l=[]
for tw in tweets:
	print(tw.text)		
	ana= TextBlob(tw.text)
	print(ana.sentiment)
	la=(tw.text, ana.sentiment)	
	l.append(la)

#stroting in a CSV file
df=pd.DataFrame(l)
df.to_csv('tweet.csv',index=False)
print('csv created succefully')

