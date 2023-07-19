import xml.etree.ElementTree as ET
import urllib.parse
import argparse
import os

def load_whitelist(whitelist_file: str):
    """
    This function reads a text file and returns a list of the lines in the file.
    """
    if not os.path.isfile(whitelist_file):
        raise FileNotFoundError(f"{whitelist_file} does not exist.")
    
    with open(whitelist_file, 'r') as file:
        return [line.strip() for line in file]

def exchange_urls(source_file: str, destination_file: str, base_url: str, whitelist: list):
    """
    This function reads an OPML file, replaces each RSS feed URL with a new URL that appends the original URL 
    as a query parameter to the base URL, and writes the resulting XML data to a new file.
    """
    # Check if source file exists
    if not os.path.isfile(source_file):
        raise FileNotFoundError(f"{source_file} does not exist.")
    
    # Load and parse the source OPML file
    try:
        source_tree = ET.parse(source_file)
        source_root = source_tree.getroot()
    except ET.ParseError as e:
        raise Exception(f"Error parsing the source file: {e}")
    
    # Traverse the entire XML tree
    for outline in source_root.iter('outline'):
        if 'xmlUrl' in outline.attrib:
            old_url = outline.attrib['xmlUrl']
            # Check if old URL is in the whitelist
            if old_url not in whitelist:
                # Encode the old URL and append it to the base URL
                # Here we remove urllib.parse.quote to avoid URL encoding
                new_url = base_url + old_url
                outline.attrib['xmlUrl'] = new_url

    # Write the new OPML file
    try:
        source_tree.write(destination_file)
    except Exception as e:
        raise Exception(f"Error writing the destination file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Exchange RSS feed URLs in an OPML file.')
    parser.add_argument('-s', '--source', required=True, help='Path to the source OPML file.')
    parser.add_argument('-d', '--destination', required=True, help='Path to the destination OPML file.')
    parser.add_argument('-b', '--baseurl', default='http://new-feed-url.com/rss?feed=', 
                        help='Base URL to prepend to the original URLs.')
    parser.add_argument('-w', '--whitelist', required=True, help='Path to a text file containing the whitelist of URLs.')
    args = parser.parse_args()

    try:
        whitelist = load_whitelist(args.whitelist)
        exchange_urls(args.source, args.destination, args.baseurl, whitelist)
        print(f"Successfully exchanged URLs in {args.source} and saved to {args.destination}")
    except Exception as e:
        print(f"Failed to exchange URLs: {e}")

if __name__ == "__main__":
    main()