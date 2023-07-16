# PodClean: Ad-free Podcasts

PodClean is a tool designed to remove ads from podcasts, providing an ad-free listening experience. It works by downloading podcasts from an RSS feed, converting speech to text, identifying ads using AI, and then removing them.

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

Then, run the application:

```bash
python podclean.py
```

## Usage

On your phone, find a Pod-catcher app that supports URL RSS feeds (e.g., Apple Podcasts even supports local RSS URLs, while Overcast only supports public RSS URLs and would require a DynDNS).

Search for the RSS feed of the podcast you want to subscribe to. Get the server address and create a URL as follows:

```
http://<server_address>:58003/rss?feed=<url_of_podcast_rss>
```

For example:

```
http://192.168.178.21:58003/rss?feed=url_of_podcast_rss
```

After subscribing, your podcatcher will be redirected to your server to get the newest episode. Please note that processing the file takes time, so expect to click on an episode and receive the processed file about the time it takes to brew a cup of coffee later.

## Workflow

1. Download Podcast from RSS-feed
2. Convert Speech to text using Whisper
3. Divide the transcript into chunks and send to ChatGPT to identify advertisements
4. Merge parts that are close to each other
5. Cut advertisement from mp3 file
6. Enjoy your ad-free podcast!

## Future Improvements

- Refactor code for better readability and performance
- Utilize Whisper API for more efficient speech-to-text conversion
- Explore open-source alternatives to OpenAI
- Create a Docker container for easier deployment
- Improve interaction with ChatGPT for better ad identification -> Testing, possiblity to flag wrong ad-marks etc.
- Add functionality to clean other content types such as triggering content, child abuse, war, etc.  -> Create your bubble... there would be even the possiblity to generate voices and let them talk what your bubble likes
- Implement a database of processed podcasts. This would only store start and end times of advertisements, allowing for efficient reprocessing of podcasts without legal concerns.

## Legal Disclaimer

This software, PodClean, is provided "as is", without any guarantees of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

PodClean uses various services and technologies to identify and remove advertisements from podcasts. The user is responsible for making sure that their use of PodClean complies with all applicable laws and regulations, and respects all third-party rights. This includes, but is not limited to, copyright law. The authors of PodClean do not endorse or encourage the violation of copyright law and cannot be held responsible for any misuse of the software.

While PodClean is designed to provide an ad-free listening experience, it does not guarantee the complete or accurate removal of all ads. It's possible for non-ad content to be mistakenly identified and removed, or for ads to be incorrectly left in.

By using PodClean, you agree to this disclaimer and use the software at your own risk.
