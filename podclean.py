import openai
import json
import math
import re
import requests
import whisper
import os
def load_api_key(secrets_file="secrets.json"):
    with open(secrets_file) as f:
        secrets = json.load(f)
    return secrets["OPENAI_API_KEY"]

api_key = load_api_key()
openai.api_key = api_key
import threading

# Create a mutex object
mutex = threading.Lock()
def spech_to_text(filename):
    # filename = os.path.basename(filename)
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

def generate_hash_key(text_segments):
    return hashlib.sha256(text_segments.encode()).hexdigest()

def ask_chatgpt(messages,num_segments=1000):
    print(num_tokens_from_string(str(messages)))
    response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=messages,
          temperature=0,
          max_tokens=num_segments*6,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

def identify_advertisement_segments(text_segments,num_segments,file,completion=False):
    messages=[
            {
              "role": "system",
              "content": f"""Act as an multilingual advertisment detecting API that recieves ordered segments of an podcast transcript: ### {text_segments} ###
Review as a whole, cluster into topics, then determine likelihood of each topic and therefore 
segment being ad/sponsored content.\n
If non-ad-segments are in between ad-segments, consider classifying them as ad.\n
Don't awnser anything but the form.\n
Only complete the likelihood in the following form:\n id ; advertisement/sponsored content (1 if >60% likely, 0 otherwise) . Only Numbers and delimiters."""
            }
          ]
    response=ask_chatgpt(messages,num_segments)
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
import ast
def num_tokens_from_messages(message_str, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a string message."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
        num_tokens = 0
        message = ast.literal_eval(message_str)
        for key, value in message.items():
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


def num_tokens_from_string(string: str, encoding_name="gpt-3.5-turbo-0613") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def process_text_segments(text_segments, file, max_retries=3, max_tokens=3696):
    """
    Processes text segments and returns cumulative results.

    Args:
        text_segments: The text segments to be processed.
        max_retries: The maximum number of retries if parsing fails.
        max_tokens: The maximum token count per request.

    Returns:
        A dictionary containing the cumulative results.
    """
    cumulative_results = {}
    batch_segments = []
    COMPLETION_TOKENS_PER_SEGMENT = 6

    msg_tokens=num_tokens_from_string(f""""role": "assistant", "content": "Act as an multilingual advertisment detecting API that recieves ordered segments of an podcast transcript: ###  ###
Review as a whole, cluster into topics, then determine likelihood of each topic and therefore 
segment being ad/sponsored content.
If non-ad-segments are in between ad-segments, consider classifying them as ad.
Don't awnser anything but the form.
Only complete the likelihood in the following form:nid ; advertisement/sponsored content (1 if >60% likely, 0 otherwise). Only Numbers and delimiters.""", "gpt-3.5-turbo")
    # print(f"Message tokens: {msg_tokens}")  # Debug print
    current_tokens = msg_tokens
    num_segments = 0
    for segment in text_segments:
        segment_text = f"{segment['id']}:{segment['text']}"
        segment_tokens=num_tokens_from_string(segment_text)
        # print(f"Segment tokens: {segment_tokens}")  # Debug print

        # If adding the current segment, its corresponding completion, and message tokens doesn't exceed the max_tokens limit
        if current_tokens + segment_tokens + COMPLETION_TOKENS_PER_SEGMENT  <= max_tokens:
            batch_segments.append(segment)
            current_tokens += segment_tokens + COMPLETION_TOKENS_PER_SEGMENT
            num_segments += 1
        else:
            # Process the current batch
            retry_count = 0
            while retry_count < max_retries:
                formatted_segments = "\n".join(f"{seg['id']}:{seg['text']}" for seg in batch_segments)
                # print(f"Formatted segments: {formatted_segments}")  # Debug print
                result_string = ""
                try:
                    result_string = identify_advertisement_segments(formatted_segments, num_segments, file)
                    # print(f"Result string tokens: {num_tokens_from_string(result_string)}")  # Debug print
                except openai.error.ServiceUnavailableError:
                    try_count += 1
                    print(f"asking ChatGPT failed...")
                    continue
                chunk_results = parse_advertisement_segments(result_string)
                for id, value in chunk_results.items():
                    if value == 1:
                        print(text_segments[id])
                if chunk_results:
                    cumulative_results.update(chunk_results)

                # Start a new batch with the current segment
                batch_segments = [segment]
                current_tokens = msg_tokens
                num_segments = 1  # Resetting num_segments for the new batch
                break  # Break out of the retry loop

    # Process the last batch
    if batch_segments:
        retry_count = 0
        while retry_count < max_retries:
            formatted_segments = "\n".join(f"{seg['id']}:{seg['text']}" for seg in batch_segments)
            # print(f"Last formatted segments: {formatted_segments}")  # Debug print
            result_string = ""
            try:
                result_string = identify_advertisement_segments(formatted_segments, num_segments, file)
                # print(f"Last result string tokens: {num_tokens_from_string(result_string)}")  # Debug print
            except:
                retry_count += 1
                print("asking ChatGPT failed. Retrying...")
                continue

            chunk_results = parse_advertisement_segments(result_string)
            if chunk_results:
                cumulative_results.update(chunk_results)
            break  # Break out of the retry loop

    return cumulative_results


def bprocess_text_segments(text_segments, file, max_retries=3, max_tokens=3900):
    """
    Processes text segments and returns cumulative results.

    Args:
        text_segments: The text segments to be processed.
        max_retries: The maximum number of retries if parsing fails.
        max_tokens: The maximum token count per request.

    Returns:
        A dictionary containing the cumulative results.
    """
    cumulative_results = {}
    batch_segments = []
    current_tokens = 0
    COMPLETION_TOKENS_PER_SEGMENT = 6

    for num_segments, segment in enumerate(text_segments):
        segment_text = f"{segment['id']}:{segment['text']}"
        segment_tokens = num_tokens_from_messages([{"role": "user", "content": segment_text}], model="gpt-3.5-turbo-0613")

        # If adding the current segment and its corresponding completion doesn't exceed the max_tokens limit
        if current_tokens + segment_tokens + COMPLETION_TOKENS_PER_SEGMENT < max_tokens:
            batch_segments.append(segment)
            current_tokens += segment_tokens + COMPLETION_TOKENS_PER_SEGMENT
        else:
            # Process the current batch
            retry_count = 0
            while retry_count < max_retries:
                formatted_segments = "\n".join(f"{seg['id']}:{seg['text']}" for seg in batch_segments)
                result_string = ""
                try:
                    result_string = identify_advertisement_segments(formatted_segments, num_segments, file)
                except Exception as e:
                    try_count += 1
                    print(f"asking ChatGPT failed Reason: {e}")
                    continue

                chunk_results = parse_advertisement_segments(result_string)
                for id, value in chunk_results.items():
                    if value == 1:
                        print(text_segments[id])
                if chunk_results:
                    cumulative_results.update(chunk_results)
                    break  # Break out of the retry loop if parsing is successful
                else:
                    retry_count += 1
                    print("Parsing failed. Retrying...")

            # Start a new batch with the current segment
            batch_segments = [segment]
            current_tokens = segment_tokens + COMPLETION_TOKENS_PER_SEGMENT

    # Process the last batch
    if batch_segments:
        retry_count = 0
        while retry_count < max_retries:
            formatted_segments = "\n".join(f"{seg['id']}:{seg['text']}" for seg in batch_segments)
            result_string = ""
            try:
                result_string = identify_advertisement_segments(formatted_segments, file)
                print(formatted_segments)
            except Exception as e:
                try_count += 1
                print(f"asking ChatGPT failed Reason: {e}")
                continue

            chunk_results = parse_advertisement_segments(result_string)
            if chunk_results:
                cumulative_results.update(chunk_results)
                break  # Break out of the retry loop if parsing is successful
            else:
                retry_count += 1
                print("Parsing failed. Retrying...")

    return cumulative_results
def aprocess_text_segments(text_segments, file, chunk_size=50, overlap_size=10, max_retries=3):
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
            result_string=""
            try:
                result_string = identify_advertisement_segments(formatted_segments,file)
                print(formatted_segments)
            except:
                retry_count += 1
                print("asking ChatGPT failed. Retrying...")
                continue
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
from flask import Flask, Response, send_file, request
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
from urllib.parse import urlencode
import hashlib

def download_and_process_podcast(mp3_url):
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

    if not os.path.exists(processed_file):
        # process podcast
        # add your processing logic here
        file_bi=spech_to_text(podcast_file)
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
    parsed_url = urlparse(old_rss_url)
    if parsed_url.scheme not in ['https']:
        abort(400, description="Invalid URL scheme: Only HTTPS is accepted")

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