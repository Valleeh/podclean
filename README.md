# podclean
Remove Ads from Podcasts
## Usage
virutalven venv
source venv/bin/activate
pip install -r requirements.txt
Create secret.json
{
    "OPENAI_API_KEY": "your_api_key"
}
python podclean.py

on your phone:
Get a Pod-catcher that supports URL RSS feeds
(Apple Podcasts support even local rss urls)
Overcast just public rss urls -> DynDNS
Search for a Rss feed of the podcast you want to subscribe to
get the address of the server
create url
Example use: http://192.168.178.21:58003/rss?feed=url_of_podcast_rss
after subscribing your podcatcher will be redirected to your server to get the newest episode. Since processing the file needs time expect to click on an episode and get the processed file about a cup of coffe later.

## Description
1. Download Podcast from RSS-feed
2. Spech to text using whispter
3. cut in chunks and send to chatgpt to identify advertisment
4. merge parts that are close to each other
5. cut advertisment from mp3
6. Profit


## Todo's:
- atlernative use whsiper API instead of local use
- clean up code
- main function
- try open source llm to get away from openai
-- take over meta data and description of podcast to new adfree feed
-- docker container for easy deployment
- improve message to chatgpt -> testing with correct identified podcasts
- Possibiliyt to clean other content then advertisment: Triggering content, Child abuse, war...
- database of proccessed pod's: just start and end times of advertisment so it can be cuted without licence problems

