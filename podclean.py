import openai
import json
import math
import re
import requests
import feedparser
import whisper
# import numpy as np
# import pandas as pd
import os
# openai.api_key = os.getenv("OPENAI_API_KEY")
def load_api_key(secrets_file="secrets.json"):
    with open(secrets_file) as f:
        secrets = json.load(f)
    return secrets["OPENAI_API_KEY"]

# Set secret API key
# Typically, we'd use an environment variable (e.g., echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc)
# However, using "internalConsole" in launch.json requires setting it in the code for compatibility with Hebrew
api_key = load_api_key()
openai.api_key = api_key
def download_podcast(pod,latest_num=1):
    # Parse the RSS feed
    feed = feedparser.parse(pod)

    # Get the latest episode
    latest_episodes = feed.entries[:latest_num]
    filenames=[]
    for episode in latest_episodes:
        # Get the URL of the mp3 file
        mp3_url = episode.enclosures[0].href

        # Get the filename
        filename = os.path.basename(mp3_url.split("?")[0])

        if os.path.exists(filename):
            print("MP3 already downloaded")
        else:
            print(f'Will download {filename}')
            # Download the mp3 file
            response = requests.get(mp3_url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f'Downloaded {filename}')
        filenames.append(filename)
    return filenames
def text_to_spech(filename):
    filename2=filename[:-4]+'.bi'
    if os.path.exists(filename2):
        print("Transcripted file exist")
    else:
        print("Transcripted file does not exist")
        model = whisper.load_model('base')
        text = model.transcribe(filename)
        with open(filename2, 'w') as file:
            json.dump(text, file)
        #printing the transcribe
        print(text['text'])
    return filename2

import json
import hashlib

JSON_FILE_PATH = "data.json"

def load_json_data(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        return {}

def save_json_data(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def generate_hash_key(text_segments):
    return hashlib.sha256(text_segments.encode()).hexdigest()

def ask_chatgpt(messages):
    print(num_tokens_from_messages(messages))
    response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=messages,
          temperature=0,
          max_tokens=2048,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]


def identify_advertisement_segments(text_segments,file,completion=False):
    data = load_json_data(file.split(".")[0]+".json")
    hash_key = generate_hash_key(text_segments)
    if hash_key in data:
        response_text = data[hash_key]
        response = openai.Response(None, {"choices": [{"text": response_text}]})
    else:
        # Query OpenAI for the response
        restart_sequence = "\n"
        if completion:
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=f"""
                Translate {text_segments} into English (if needed). Review as a whole, cluster into topics, then determine likelihood of each being ad/sponsored content.
                \n
                Please give the likelihood in the following form:\n
                id ; advertisement/sponsored content (1 if >60% likely, 0 otherwise)
                \n
                """,
                temperature=0,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            data[hash_key] = response.choices[0].text
        else:
            messages=[
                    {
                      "role": "system",
                      "content": f"""Translate the following segments into English (if needed).\n
                        {text_segments}\n
                        Review as a whole, cluster into topics, then determine likelihood of each topic and therefore segment being ad/sponsored content.\n
                        If non-ad-segments are in between ad-segments, consider classifying them as ad.\n

                        Don't awnser anything but the form.\n
                        Only complete the likelihood in the following form:\nid ; advertisement/sponsored content (1 if >60% likely, 0 otherwise). Only Numbers and delimiters."""
                    }
                  ]
            data[hash_key]=ask_chatgpt(messages)
            response=data[hash_key]
                    # Cache the question and response in the JSON file
        
        save_json_data(data, JSON_FILE_PATH)
    print(response)
    return response
def pre_process_text(filename):
    with open(filename, 'r') as file:
        content = json.load(file)
    return [{'id':item['id'],'text':item['text']} for item in content['segments']],content
def parse_advertisement_segments(result_string):
    """
    Parses a string of identified segments and returns a dictionary.

    Args:
        result_string: The string to be parsed.

    Returns:
        A dictionary containing the parsed segments.
    """
    parsed_results = {}
    lines = result_string.split("\n")
    
    for line in lines:
        if not line:
            continue
        try:
            segment_id, segment_value = re.split(';|:', line)
            parsed_results[int(segment_id.strip())] = int(segment_value.strip())
        except ValueError:
            print(f"Error parsing line: {line}")
    return parsed_results

import tiktoken
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        
def process_text_segments(text_segments, file, chunk_size=70, overlap_size=10, max_retries=3):
    """
    Processes text segments in chunks and returns cumulative results.
    Includes overlap between chunks to maintain context.

    Args:
        text_segments: The text segments to be processed.
        chunk_size: The size of each chunk.
        overlap_size: The size of the overlap between chunks.
        max_retries: The maximum number of retries if parsing fails.

    Returns:
        A dictionary containing the cumulative results.
    """
    total_segments = len(text_segments)
    num_chunks = math.ceil((total_segments - overlap_size) / (chunk_size - overlap_size))
    cumulative_results = {}

    for chunk in range(num_chunks):
        start_index = chunk * (chunk_size - overlap_size)
        end_index = min(start_index + chunk_size, total_segments)
        chunked_text_segments = text_segments[start_index:end_index]
        formatted_segments = "\n".join(f"{segment['id']}:{segment['text']}" for segment in chunked_text_segments)

        retry_count = 0
        while retry_count < max_retries:
#             print(formatted_segments)
            result_string = identify_advertisement_segments(formatted_segments,file)
            chunk_results = parse_advertisement_segments(result_string)
            if chunk_results:
                cumulative_results.update(chunk_results)
                break  # Break out of the retry loop if parsing is successful
            else:
                retry_count += 1
                print("Parsing failed. Retrying...")

    return cumulative_results
def ad_segments_to_times(ad_segments,content):
    start_times = []
    end_times = []

    segments=content['segments']
    # filtered_segments = [segment for segment in segments if result_dict[int(segment['id'])]]

    filtered_segments = []
    for segment in segments:
        segment_id = int(segment['id'])
        if segment_id in ad_segments and ad_segments[segment_id]:
            filtered_segments.append(segment)

#     print(filtered_segments)
    for segment in filtered_segments:
        start_times.append(segment['start'])
        end_times.append(segment['end'])

    merged_start_times = []
    merged_end_times = []

    if len(start_times) == len(end_times):
        merged_start_times.append(start_times[0])
        current_end = end_times[0]

        for i in range(1, len(start_times)):
            if start_times[i] <= current_end:
                current_end = max(current_end, end_times[i])
            else:
                merged_end_times.append(current_end)
                merged_start_times.append(start_times[i])
                current_end = end_times[i]

        merged_end_times.append(current_end)

#     print(merged_start_times)  # Output: [205.0, 274.0, 310.0, 364.0]
#     print(merged_end_times)  # Output: [211.0, 288.0, 321.0, 386.0]
    return merged_start_times,merged_end_times

from pydub import AudioSegment 
def remove_ads(input_file, output_file, start_times, end_times):
    """
    Removes sections from an audio file.
    
    :param input_file: Path to the input audio file.
    :param output_file: Path to save the output audio file.
    :param start_times: List of start times in seconds.
    :param end_times: List of end times in seconds.
    """
    print("Loading MP3 file")
    # Load MP3 file
    try:
        audio = AudioSegment.from_mp3(input_file)
    except Exception as e:
        print("Error loading MP3 file:", e)
        return
    
    print("File loaded, duration: {} ms".format(len(audio)))
    
    # Ensure start_times and end_times have the same length
    if len(start_times) != len(end_times):
        print("Error: start_times and end_times must have the same length")
        return
    
    # Convert start_times and end_times to milliseconds
    start_times = [int(x * 1000) for x in start_times]
    end_times = [int(x * 1000) for x in end_times]
    
    # Cut out the ads
    print("Removing ads")
    final_audio = AudioSegment.empty()
    last_end = 0
    for start, end in zip(start_times, end_times):
        if start < last_end or end < start:
            print("Error: Time values are incorrect, start should be less than end and segments should not overlap")
            return
        print("Keeping audio from {} to {} ms".format(last_end, start))
        final_audio += audio[last_end:start]
        last_end = end
    print("Keeping audio from {} to end".format(last_end))
    final_audio += audio[last_end:]
    
    # Save the result to a new MP3 file
    print("Exporting final audio to {}".format(output_file))
    try:
        final_audio.export(output_file, format="mp3")
        print("File exported successfully")
    except Exception as e:
        print("Error exporting file:", e)

def merge_time_segments(start_times, end_times, min_duration=5.0, max_gap=20.0):
    """
    Merge time segments that are close to each other and filter out segments shorter than the minimum duration.
    
    :param start_times: List of start times
    :param end_times: List of end times
    :param min_duration: Minimum duration for a segment to be included
    :param max_gap: Maximum gap between segments to merge them
    :return: Two lists representing the start times and end times of the final segments
    """
    if len(start_times) != len(end_times):
        raise ValueError("start_times and end_times must have the same length")

    merged_start_times = []
    merged_end_times = []

    # Set the initial segment
    current_start = start_times[0]
    current_end = end_times[0]

    for i in range(1, len(start_times)):
        # If the gap between current end and next start time is smaller or equal to max_gap, extend the segment
        if start_times[i] - current_end <= max_gap:
            current_end = end_times[i]
        else:
            # If not, check if the current segment is long enough to keep it
            if current_end - current_start >= min_duration:
                merged_start_times.append(current_start)
                merged_end_times.append(current_end)
            # Start a new segment
            current_start = start_times[i]
            current_end = end_times[i]

    # Check the last segment
    if current_end - current_start >= min_duration:
        merged_start_times.append(current_start)
        merged_end_times.append(current_end)

    return merged_start_times, merged_end_times
from flask import Flask, Response
from feedgen.feed import FeedGenerator
import feedparser

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
# def parse_feed(feed_url, new_mp3_url):
#     d = feedparser.parse(feed_url)

#     fg = FeedGenerator()
#     fg.load_extension('podcast')
#     fg.link(href=SERVER_HOME)
#     fg.podcast.itunes_author('Cory Doctorow')
#     fg.podcast.itunes_category([
#         {'cat': 'Book', 'sub': 'Technology'},
#         {'cat': 'Book', 'sub': 'Science Fiction'}
#     ])
#     fg.podcast.itunes_complete('yes')
#     fg.podcast.itunes_image('/'.join([SERVER_HOME, 'image/radicalized-full-US-cover-large.jpg']))
#     fg.title('radicalized')
#     fg.podcast.itunes_subtitle('Four tales of our present moment')
#     fg.description("""If you want a better future tomorrow, you're going to have to fight for it today.
#     Here are four urgent stories from author and activist Cory Doctorow, four social, technological and economic visions of the world today and its near – all too near – future.
#     'Unauthorized Bread' is a tale of immigration, toxic economic stratification and a young woman's perilously illegal quest to fix a broken toaster.
#     In 'Model Minority' a superhero finds himself way out his depth when he confronts the corruption of the police and justice system.
#     'Radicalized' is the story of a desperate husband, a darknet forum and the birth of a violent uprising against the US health care system.
#     The final story, 'The Masque of the Red Death', tracks an uber-wealthy survivalist and his followers as they hole up and attempt to ride out the collapse of society.""")
#     for entry in d.entries:
#         fe = fg.add_entry()
#         fe.id(entry.link)
#         fe.title(entry.title)
#         fe.description(entry.description)
#         fe.enclosure(url=new_mp3_url, type='audio/mpeg')

#     rssfeed = fg.rss_str(pretty=True)
#     return rssfeed

#@app.route('/')
#def serve_feed():
#    feed_url="https://feeds.feedburner.com/HacksOnTap"
#    # filenames=download_podcast('https://pythonbytes.fm/episodes/rss')
#    # filenames=download_podcast('https://feeds.megaphone.fm/hubermanlab')
#    #filenames=download_podcast('https://feeds.simplecast.com/54nAGcIl')
#    # filenames=download_podcast('https://rss.wbur.org/onpoint/podcast')
#    #filenames=download_podcast('https://changelog.com/podcast/feed')
#    # filenames=download_podcast('https://feeds.lagedernation.org/feeds/ldn-mp3.xml')
#    filenames=download_podcast(feed_url)
#    #filenames=download_podcast("https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/f5d5fac6-77be-47e6-9aee-ae32006cd8c3/b26cbbeb-86eb-4b97-9b34-ae32006cd8d6/podcast.rss")
     
#    # for file in filenames:
#    file=filenames[0]
#    output_file = file.split(".")[0]+"_adfree.mp3"
#    if os.path.exists(file):
#        new_feed = parse_feed(feed_url, output_file)
#        response = Response(new_feed, mimetype='application/rss+xml')
#        return response

#    file_bi=text_to_spech(file)

#    text_segments,content = pre_process_text(file_bi)
#    results_dict = process_text_segments(text_segments,file)
#    print(results_dict)

#    merged_start_times,merged_end_times=ad_segments_to_times(results_dict, content)
#    final_start_times, final_end_times=merge_time_segments(merged_start_times,merged_end_times, min_duration=5.0, max_gap=15.0)
#    print(final_start_times)
#    print(final_end_times)

#    print(file)

#    remove_ads(file, output_file, final_start_times, final_end_times)  
#    new_feed = parse_feed(feed_url, output_file)
#    response = Response(new_feed, mimetype='application/rss+xml')
#    return response
from urllib.parse import urlencode
import hashlib
from flask import Flask, send_file, request

PODCAST_FOLDER = 'podcasts'
PROCESSED_FOLDER = 'processed'

def get_podcast_file(mp3_url):
    # calculate a hash of the URL to use as filename
    filename = hashlib.md5(mp3_url.encode()).hexdigest()
    return os.path.join(PODCAST_FOLDER, filename)

def get_processed_file(mp3_url):
    # calculate a hash of the URL to use as filename
    filename = hashlib.md5(mp3_url.encode()).hexdigest()
    return os.path.join(PROCESSED_FOLDER, filename)

def download_and_process_podcast(mp3_url):
    # podcast_file = get_podcast_file(mp3_url)
    # processed_file = get_processed_file(mp3_url)
    podcast_file = 'podcasts/' + hashlib.md5(mp3_url.encode()).hexdigest()
    processed_file = podcast_file + '_processed.mp3'

    # Check if the processed file already exists
    if not os.path.exists(processed_file):
        # If the processed file doesn't exist, check if the podcast needs to be downloaded
        if not os.path.exists(podcast_file):
            # Ensure the directory exists before downloading
            os.makedirs(os.path.dirname(podcast_file), exist_ok=True)
            
            # Download the podcast
            response = requests.get(mp3_url, stream=True)
            response.raise_for_status()
            with open(podcast_file, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)   
    # if not os.path.exists(podcast_file):
    #     # download podcast
    #     response = requests.get(mp3_url, stream=True)
    #     response.raise_for_status()
    #     with open(podcast_file, 'wb') as file:
    #         for chunk in response.iter_content(chunk_size=8192):
    #             file.write(chunk)

    if not os.path.exists(processed_file):
        # process podcast
        # add your processing logic here
        # output_file = file.split(".")[0]+"_adfree.mp3"
        # if os.path.exists(file):
        #     new_feed = parse_feed(feed_url, processed_file)
            # response = Response(new_feed, mimetype='application/rss+xml')
            # return response
        file_bi=text_to_spech(podcast_file)
        text_segments,content = pre_process_text(file_bi)
        results_dict = process_text_segments(text_segments,podcast_file)
        print(results_dict)

        merged_start_times,merged_end_times=ad_segments_to_times(results_dict, content)
        final_start_times, final_end_times=merge_time_segments(merged_start_times,merged_end_times, min_duration=5.0, max_gap=15.0)
        print(final_start_times)
        print(final_end_times)
        print(podcast_file)
        remove_ads(podcast_file, processed_file, final_start_times, final_end_times)  
        print("Processsing finished")

    return processed_file

@app.route('/podcast')
def serve_podcast():
    # get podcast url from query params
    mp3_url = request.args.get('url')
    
    processed_file = get_processed_file(mp3_url)
    if not os.path.exists(processed_file):
        processed_file = download_and_process_podcast(mp3_url)
    
    return send_file(processed_file, mimetype='audio/mpeg')

from feedgen.feed import FeedGenerator

def generate_new_feed(feed):
    fg = FeedGenerator()
    fg.title(feed.feed.title)
    fg.link(href=feed.feed.link, rel='alternate')
    fg.description(feed.feed.description)

    for entry in feed.entries:
        fe = fg.add_entry()
        fe.title(entry.title)
        fe.link(href=entry.enclosures[0].href)
        fe.description(entry.description)

    return fg.rss_str(pretty=True)
@app.route('/rss')
def serve_rss():
    # get old rss feed url from query params
    old_rss_url = request.args.get('feed')

    # parse old rss feed
    old_feed = feedparser.parse(old_rss_url)

    fg = FeedGenerator()
    fg.title(old_feed.feed.title)
    fg.link(href=old_feed.feed.link, rel='alternate')
    fg.description(old_feed.feed.description)

    # replace mp3 urls with our server's podcast endpoint
    for entry in old_feed.entries:
        fe = fg.add_entry()
        fe.title(entry.title)
        fe.link(href=entry.link)
        fe.description(entry.description)
        
        # Redirect enclosure href to our podcast endpoint
        mp3_url = entry.enclosures[0].href
        new_mp3_url = f'{request.url_root}podcast?{urlencode({"url": mp3_url})}'
        fe.enclosure(new_mp3_url, 0, 'audio/mpeg')

    # generate new rss feed
    new_feed = fg.rss_str(pretty=True)

    return Response(new_feed, mimetype='application/rss+xml')

@app.route('/index.xml')
def get_feed():
    return Response(fg.rss_str(), mimetype='application/rss+xml')
if __name__ == '__main__':
    app.run(host='192.168.178.32', port=58003)
