# podclean
Remove Ads from Podcasts
## Description
1. Download Podcast from RSS-feed
2. Spech to text using whispter
3. cut in chunks and send to chatgpt to identify advertisment
4. merge parts that are close to each other
5. cut advertisment from mp3


## Todo's:
- try open source llm to get away from openai
- establish a podcast feed after proccesed files
-- take over meta data and description of podcast to new adfree feed
-- docker container for easy deployment
- improve message to chatgpt -> testing with correct identified podcasts
- Possibiliyt to clean other content then advertisment: Triggering content, Child abuse, war...
- database of proccessed pod's: just start and end times of advertisment so it can be cuted without licence problems

