# PodClean: Ad-free Podcasts

PodClean is a tool that aims to provide an option for ad-free podcast listening. It achieves this through an intermediary server that fakes podcast RSS feeds, facilitating the downloading and processing of podcast episodes.

The processing of podcasts involves converting the speech to text, identifying ads using OpenAI API, and cuting them from the MP3. This intermediary server-based architecture can be integrated into existing podcast listening workflows, providing an option for users who prefer an ad-free experience.
## Installation

First, setup your Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `secret.json` file in the project root directory, and include your OpenAI API Key:

```json
{
    "OPENAI_API_KEY": "your_api_key"
}
```
how to get an OPENAI-API key -> https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/

Then, run the application:

```bash
python server.py --port <port>  
```
## Server

Python Flask application designed to work as an intermediary server for podcast RSS feeds. 

1. __Podcast Processing Endpoint__ (`/podcast`): A GET route that takes a URL to a podcast episode (MP3 file) as a query parameter. It validates the URL, downloads and processes the podcast if it doesn't exist on the server, and then serves the processed podcast. The application also uses a threading lock to prevent multiple simultaneous downloads and processing of the same podcast.

2. __RSS Feed Endpoint__ (`/rss`): A GET route that takes a URL to a podcast RSS feed as a query parameter. It fetches the original RSS feed, iterates through each enclosure (podcast episode), and modifies the URL to point to the `/podcast` route on this server, effectively "overriding" the original MP3 URLs with URLs that point to this server. The application employs a whitelist to exclude certain feeds from this URL modification.

Additionally, the application includes a helper function to retrieve the IP address of the server. It starts by listening on `0.0.0.0:<port>`, making it accessible from any IP address.

## Usage

On your phone, find a Pod-catcher app that supports URL RSS feeds (e.g., Apple Podcasts even supports local RSS URLs, while Overcast only supports public RSS URLs and would require DynDNS or an actual server).

Search for the RSS feed of the podcast you want to subscribe to. Get the server address and create a URL as follows:

```
http://<server_address>:<port>/rss?feed=<url_of_podcast_rss>
```

For example:

```
http://192.168.178.21:8642/rss?feed=url_of_podcast_rss
```

After subscribing, your podcatcher will be redirected to your server to get the newest episode. Please note that processing the file takes time, so expect to click on an episode and receive the processed file about the time it takes to brew a cup of coffee later. 

:warning: **IMPORTANT: Each use of ChatGPT costs around $0.04 per podcast. Please keep this in mind when using PodClean!**

Make use of a whitelist.txt so that podcasts without any advertisements are not unnecessarily processed.

## OPML Transformation  
OPML files are widely recognized as the standard format for exporting and importing data among podcatchers.

Using opml_transform.py, you can modify the associated RSS feeds within the OPML file to redirect to your server.

```bash
python opml_transform.py -s overcast.opml -d over.opml -b "http://your-server-adress:<port>/rss?feed=" --w whitelist.txt

```
This process essentially allows you to automatically adjust all your favorite podcast feeds to become ad-free via the PodClean tool.

## Processing Steps

1. Download Podcast given in the RSS-feed
2. Convert Speech to text using Whisper
3. Divide the transcript into chunks and send to ChatGPT to identify advertisements
4. Merge parts that are close to each other
5. Cut advertisement from mp3 filek
6. Enjoy your ad-free podcast!

## Future Improvements / ToDo's

- Refactor code for better readability and performance
- https support and some URL's are wrong
- Utilize Whisper API for more efficient speech-to-text conversion -> higher cost(approx 1-2$ per podcast), could run on a raspi
- Explore open-source alternatives to OpenAI -> e.g. https://huggingface.co/morenolq/spotify-podcast-advertising-classification
- Create a Docker container for easier deployment
- Improve interaction with ChatGPT for better ad identification -> Testing, possiblity to flag wrong ad-marks etc.
- Add functionality to clean other content types such as triggering content, child abuse, war, etc.  -> Create your bubble... there would be even the possiblity to generate voices and let them talk what your bubble likes
- Implement a database of processed podcasts. This would only store start and end times of advertisements, allowing for efficient reprocessing of podcasts without legal concerns.
- Split Server's into one "Podcast Content and Feed Management Server" and one "Processing server"

## Legal Disclaimer

This software, PodClean, is provided "as is", without any guarantees of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

PodClean uses various services and technologies to identify and remove advertisements from podcasts. The user is responsible for making sure that their use of PodClean complies with all applicable laws and regulations, and respects all third-party rights. This includes, but is not limited to, copyright law. The authors of PodClean do not endorse or encourage the violation of copyright law and cannot be held responsible for any misuse of the software.

While PodClean is designed to provide an ad-free listening experience, it does not guarantee the complete or accurate removal of all ads. It's possible for non-ad content to be mistakenly identified and removed, or for ads to be incorrectly left in.

By using PodClean, you agree to this disclaimer and use the software at your own risk.
