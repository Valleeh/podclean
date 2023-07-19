from flask import Flask, Response, send_file, request
from podclean import download_and_process_podcast
import os
import feedparser
import threading
# Create a mutex object
mutex = threading.Lock()

app = Flask(__name__)
import socket
def get_ip_address():
    """Get current IP address.
    From https://stackoverflow.com/a/166589/379566
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]
SERVER_HOME = 'http://{}'.format(get_ip_address())
from urllib.parse import urlencode
import hashlib


@app.route('/podcast')
def serve_podcast():
    # get podcast url from query params
    mp3_url = request.args.get('url')

    if mp3_url is None:
        abort(400, description="No URL provided")

    # parse and check url scheme
    parsed_url = urlparse(mp3_url)
    if parsed_url.scheme not in ['https']:
        abort(400, description="Invalid URL scheme: Only HTTPS is accepted")
    podcast_file = 'podcasts/' + hashlib.md5(mp3_url.encode()).hexdigest()
    processed_file = podcast_file + '_processed.mp3'
    print(processed_file)
    if not os.path.exists(processed_file):
        print("Process podcast")
        mutex.acquire()
        try:
            if os.path.exists(processed_file):
                return send_file(processed_file, mimetype='audio/mpeg')
            processed_file = download_and_process_podcast(mp3_url)
        finally:
            mutex.release()
    print(processed_file)
    return send_file(processed_file, mimetype='audio/mpeg')



from flask import request, abort, Response
from urllib.parse import urlparse, urlencode
import requests
from lxml import etree
@app.route('/rss')
def serve_rss():
    # get old rss feed url from query params
    old_rss_url = request.args.get('feed')

    # read the whitelist from a file, each url on a separate line
    try:
        with open('whitelist.txt', 'r') as f:
            whitelist = [line.strip() for line in f.readlines()]
    except IOError:
        abort(500, description="Cannot read whitelist file")

    if old_rss_url is None:
        abort(400, description="No feed URL provided")

    # parse and check url scheme
    # parsed_url = urlparse(old_rss_url)
    # if parsed_url.scheme not in ['https']:
    #     abort(400, description="Invalid URL scheme: Only HTTPS is accepted")

    try:
        # Fetch the original RSS feed
        response = requests.get(old_rss_url)
        response.raise_for_status()  # Raises a HTTPError if the response status isn't 200
    except requests.HTTPError as http_err:
        abort(400, description=f"HTTP error occurred: {http_err}")
    except Exception as err:
        abort(500, description=f"Other error occurred: {err}")

    try:
        root = etree.fromstring(response.content)
    except etree.ParseError as err:
        abort(400, description=f"Error parsing the XML: {err}")

    # Only change the MP3 URL(s) if the feed is not in the whitelist
    if old_rss_url not in whitelist:
        for enclosure in root.xpath('//enclosure'):
            mp3_url = enclosure.get('url')
            new_mp3_url = f'{request.url_root}podcast?{urlencode({"url": mp3_url})}'
            enclosure.set('url', new_mp3_url)

    new_feed = etree.tostring(root, xml_declaration=True, encoding='utf-8')

    return Response(new_feed, mimetype='application/rss+xml')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=58003)
