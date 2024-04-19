import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import quote, unquote
import re

"""
This script scrapes poems from the website dikt.org and saves them as JSON files.
It fetches all category links from the main page, then iterates through each category to get poem links.
For each poem link, the script fetches the poem text and saves it as a JSON file.
If a poem with the same title already exists, the script updates the poem text and tags.
"""

# Base URL
BASE_URL = "https://dikt.org"

# Function to construct a full URL from a relative path
def construct_url(path):
    if not path:
        return None
    # Decode path first to handle any encoded characters
    path = unquote(path)
    if "dikt.org" not in path and not path.startswith("/"):
        print(f"Skipping non-dikt.org URL: {path}")
        return None
    # Encode only after cleaning the path
    encoded_path = quote(path, safe='/:')
    if path.startswith('http'):
        return encoded_path
    elif path.startswith('/'):
        return BASE_URL + encoded_path
    else:
        return BASE_URL + '/' + encoded_path

# Function to fetch a BeautifulSoup object from a URL
def fetch_soup(url):
    try:
        response = requests.get(url)
        content_type = response.headers.get('Content-Type', '')

        # Check if the response is HTML, otherwise, skip parsing
        if 'text/html' in content_type:
            response.encoding = 'utf-8'  # Ensures that text is correctly interpreted as UTF-8
            return BeautifulSoup(response.content, 'html.parser')
        else:
            print(f"Skipping non-HTML content at {url}: {content_type}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_category_link(base_url, skip_categories):
    soup = fetch_soup(base_url)
    results = soup.find(id="mw-subcategories")
    paths = []
    # Skip categories that are not relevant
    if results:
        links = results.find_all("a")
        for link in links:
            path = link.get('href')
            if path:  # Ensure path is not None before processing
                full_path = construct_url(path)
                if full_path and not any(skip_cat == path.split('/Kategori:')[-1] for skip_cat in skip_categories): 
                    paths.append(full_path)
    return paths

# Function to get all category links from the main page
def get_all_category_links(base_url):
    soup = fetch_soup(base_url)
    results = soup.find(id="n-dikt-etter-emne") # Find the div containing the category links
    skip_categories = ["Alle_dikt_alfabetisk", "Dikt_av_ukjent_forfatter", "Dikt_etter_emne", "Dikt_etter_århundre", "Dikt_på_Engelsk", "Diktsamlinger", "Forfattere"] # Skip these categories
    category_links = []
    if results:
        links = results.find_all("a")
        for link in links:
            category_url = link.get('href')
            if category_url:
                # Ensure the URL is decoded before processing
                decoded_url = unquote(category_url)
                category_part = decoded_url.split('/Kategori:')[-1]
                clean_category = category_part.strip()
                if not any(clean_category == skip_cat for skip_cat in skip_categories):
                    full_category_url = construct_url(decoded_url)
                    subcategory_links = get_category_link(full_category_url, skip_categories)
                    category_links.extend(subcategory_links if subcategory_links else [full_category_url])
                else:
                    print(f"Skipping category: {clean_category}")
    return category_links


# Function to get all poem paths from a category page
def get_poem_paths(poem_category_url):
    soup = fetch_soup(poem_category_url)
    if not soup:
        return []
    results = soup.find("div", class_="mw-content-ltr") # Find the div containing the poem links
    poem_paths = []
    if results:
        links = results.find_all("a")
        for link in links:
            poem_path = link.get("href")
            constructed_url = construct_url(poem_path)
            if constructed_url:  # Only append if URL is not None
                poem_paths.append(constructed_url)
    return poem_paths

# Function to get poem data from a poem URL
def get_poem_data(poem_url):
    soup = fetch_soup(poem_url)
    if not soup:
        print(f"No soup could be created for the URL: {poem_url}")
        return None

    poem_div = soup.find("div", class_="dikt")
    tag_div = soup.find(id="mw-normal-catlinks")
    poem_data = {}
    if poem_div:
        # Replace <br> or new line tags with \n
        poem_text = []
        for element in poem_div.find_all('p'):
            for content in element.contents:
                if isinstance(content, str):
                    poem_text.append(content)
                elif content.name == 'br':
                    poem_text.append('\n')
        # Join all parts of the poem into a single string, handling line breaks correctly.
        poem_data['poem'] = ''.join(poem_text).strip()
    else:
        print(f"No poem content found for the URL: {poem_url}")
        return None

    # Extract tags if the tag div exists
    if tag_div:
        poem_data['tags'] = [li.get_text(strip=True) for li in tag_div.find_all("li")]

    return poem_data

# Function to save poem data to a JSON file
def save_poem(poem_data, title):
    safe_title = unquote(title).replace('/', ' ').replace('\\', ' ').replace(':', '-')
    file_path = os.path.join('poems', f"{safe_title}.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(poem_data, file, ensure_ascii=False, indent=4)
    print(f"Saved poem to {file_path}")

# Function to process a poem URL
def process_poem(link):
    poem_data = get_poem_data(link)
    if not poem_data:
        return
    
    poem_title = link.split('/')[-1]
    safe_title = unquote(poem_title).replace('/', ' ').replace('\\', ' ').replace(':', '-')
    file_path = os.path.join('poems', f"{safe_title}.json")

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            print(f"File is empty, skipping update: {file_path}") # Skip empty files
            return
        with open(file_path, 'r+', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError: # Handle JSON decoding errors
                print(f"Error reading from JSON file. It might be corrupt or incomplete: {file_path}")
                return
            # Merge tags and update poem text if different
            existing_tags = set(existing_data.get('tags', []))
            new_tags = set(poem_data.get('tags', []))
            all_tags = list(existing_tags.union(new_tags))

            if existing_data.get('poem') != poem_data.get('poem'):
                existing_data['poem'] = poem_data.get('poem')

            existing_data['tags'] = all_tags
            file.seek(0)
            json.dump(existing_data, file, ensure_ascii=False, indent=4)
            file.truncate()  # Make sure to cut off the extra data if any
        print(f"Updated poem: {file_path}")
    else:
        save_poem(poem_data, poem_title)

def add_title_to_json(directory):
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            # Remove '.json' and replace underscores with spaces to create the title
            title = filename[:-5].replace('_', ' ')
            
            # Open the JSON file, load its content, and modify it
            with open(file_path, 'r+', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    # Check if 'title' already exists to avoid overwriting it
                    if 'title' not in data:
                        data['title'] = title
                        # Write the modified data back to the file
                        file.seek(0)  # Go back to the start of the file
                        json.dump(data, file, ensure_ascii=False, indent=4)
                        file.truncate()  # Remove remaining part of old content
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON from {file_path}")
                except Exception as e:
                    print(f"An error occurred while processing {file_path}: {e}")

def main():
    category_links = get_all_category_links(base_url=BASE_URL)
    
    # Download poems
    for link in category_links:
        poem_paths = get_poem_paths(link)
        for poem_link in poem_paths:
            process_poem(poem_link)

    # Add title to JSON files
    directory_path = 'poems'
    add_title_to_json(directory_path) 

if __name__ == "__main__":
    main()
