import streamlit as st
import torch
import yt_dlp
from pytube import Playlist
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import io
import os
import re
import cv2
from yt_dlp import YoutubeDL
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


# Function to convert cookies in JSON or TXT format to Netscape format
def convert_cookies_to_netscape(cookies_file):
    try:
        if cookies_file.name.endswith('.json'):
            cookies = json.load(cookies_file)
        elif cookies_file.name.endswith('.txt'):
            cookies = []
            for line in cookies_file.readlines():
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    cookies.append({
                        'domain': parts[0],
                        'name': parts[5],
                        'value': parts[6],
                        'path': parts[2],
                        'expirationDate': parts[4] if len(parts) > 4 else None
                    })
        else:
            raise ValueError("Invalid cookie file format. Only JSON or TXT are supported.")

        # Convert cookies to Netscape format
        netscape_cookies = []
        for cookie in cookies:
            expiry = cookie.get('expiry', '')
            if expiry:
                expiry = str(int(expiry))
            netscape_cookies.append(
                f"{cookie['domain']}\tTRUE\t{cookie['path']}\t{expiry}\t{cookie['name']}\t{cookie['value']}"
            )

        return netscape_cookies

    except Exception as e:
        st.error(f"Error converting cookies: {str(e)}")
        return None



# Function to download a YouTube video using yt-dlp and a cookie file
def download_video(url, cookie_file=None):
    download_status = ""  # Initialize download_status to avoid referencing undefined variable
    downloaded_video_path = None

    # Define the directory to store the video
    temp_dir = "temp_video"  # Name of the temporary directory

    # Create the directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Create the directory

    # Set up yt-dlp options, saving video to the temp directory
    ydl_opts = {
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),  # Save video to the temp directory
    }

    # Add cookie file to yt-dlp options if provided
    if cookie_file:
        ydl_opts['cookiefile'] = cookie_file

    # Use yt-dlp to download the video
    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            download_status = "Video downloaded successfully!"  # Set the success status
            # Prepare the full path of the downloaded video (fixing path issue)
            downloaded_video_path = os.path.join(f"{ydl.prepare_filename(ydl.extract_info(url, download=False))}")
            st.success(download_status)
        except Exception as e:
            download_status = f"Error downloading video: {str(e)}"  # Set the error message
            st.error(download_status)  # Display error message

    print(downloaded_video_path)
    return download_status, downloaded_video_path  # Return the status for further checking if needed


# Function to get video URLs from multiple playlists or individual video links
def get_video_urls_multiple(input_urls):
    video_urls = []
    urls = input_urls.split(",")  # Split input by comma
    for url in urls:
        url = url.strip()  # Remove any leading/trailing spaces
        if "playlist" in url:
            playlist = Playlist(url)
            video_urls.extend(playlist.video_urls)  # Add all video URLs in the playlist
        else:
            video_urls.append(url)  # Treat as a single video URL
    return video_urls


# Function to get transcript for a video using its YouTube ID
def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        # Fetch the transcript (if available)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        return None


# Function to check if a video is under Creative Commons license using YouTube Data API and description

# Function to format the transcript into a readable form
def format_transcript(transcript):
    formatted_transcript = []
    for entry in transcript:
        start_time = entry['start']  # Timestamp
        duration = entry['duration']
        text = entry['text']  # Transcript text
        formatted_transcript.append(f"[{start_time}s - {start_time + duration}s] {text}")
    return formatted_transcript


# Function to process input (multiple playlists or individual videos) and fetch transcripts for all videos
def process_input(input_urls):
    video_urls = get_video_urls_multiple(input_urls)
    if not video_urls:
        return []

    all_transcripts = []  # List to hold all transcripts

    video_chunks = {}  # Dictionary to store video-specific transcripts

    # Use another ThreadPoolExecutor to fetch transcripts concurrently
    with ThreadPoolExecutor() as transcript_executor:
        future_to_video = {transcript_executor.submit(get_transcript, video_url): video_url for video_url in video_urls}
        for idx, future in enumerate(as_completed(future_to_video)):
            video_url = future_to_video[future]
            try:
                transcript = future.result()
                if transcript:
                    formatted_transcript = format_transcript(transcript)
                    video_chunks[video_url] = formatted_transcript  # Store by video URL
                else:
                    video_chunks[video_url] = ["Transcript not available"]
            except Exception as e:
                video_chunks[video_url] = ["Transcript extraction failed"]
                print(f"Error getting transcript for {video_url}: {e}")

    # Reassemble the output in the original order of video URLs
    for video_url in video_urls:
        all_transcripts.append(
            {"video_url": video_url, "transcript": video_chunks.get(video_url, ["No transcript found"])})
    return all_transcripts


def process_query(query, stored_transcripts, threshold=0.3):
    if not query:
        print("Please enter a query to search in the transcripts.")
        return []

    if not stored_transcripts:
        print("No transcripts available. Please process a playlist or video first.")
        return []

        # Function to identify question type (what, why, how, etc.)
    def get_question_type(query):
        query = query.lower()
        if query.startswith("what"):
            return "what"
        elif query.startswith("why"):
            return "why"
        elif query.startswith("how"):
            return "how"
        else:
            return "general"

    # Determine the question type to adjust processing
    question_type = get_question_type(query)

    # Adjust the threshold based on question type for better relevance
    if question_type == "what":
        threshold = 0.4  # Slightly lower threshold for "what" questions
    elif question_type == "why":
        threshold = 0.5  # Higher threshold for "why" questions (more context needed)
    elif question_type == "how":
        threshold = 0.5  # Higher threshold for "how" questions
    # Use tokenizer to encode the query
    inputs = tokenizer.encode(query, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    # Prepare the list of transcript lines with their timestamps
    all_transcripts_text = []
    transcript_lines = []
    timestamps = []
    video_urls = []

    for video in stored_transcripts:
        video_url = video['video_url']
        if isinstance(video['transcript'], list):
            for line in video['transcript']:

                if isinstance(line, str):
                    # Example line format: "[2.03s - 4.54s] some text"
                    parts = line.split(']')  # Split at the closing bracket
                    time_text = parts[0].strip('[')  # Extract the time part by removing '['
                    text = parts[1].strip()  # The remaining part is the text

                    # Extract start_time and end_time from time_text
                    time_parts = time_text.split('-')  # Split at the dash
                    try:
                        start_time = float(time_parts[0].replace('s', '').strip())  # Convert to float and remove 's'
                        end_time = float(time_parts[1].replace('s', '').strip())  # Convert to float and remove 's'
                    except ValueError:
                        # Default to 0.0 if conversion fails
                        start_time = 0.0
                        end_time = 0.0

                    # Create a dictionary for the line
                    line = {'text': text, 'start_time': start_time, 'end_time': end_time}
                # Check if the transcript line is a dictionary with 'text', 'start_time', and 'end_time'

                if isinstance(line, dict) and 'text' in line and 'start_time' in line and 'end_time' in line:
                    text = line['text']
                    start_time = line['start_time']  # Ensure it's a float if it's not
                    end_time = line['end_time']

                else:
                    # If the line does not match the expected format, assign default values
                    start_time = 0.0
                    end_time = 0.0
                    text = line  # The entire line is treated as text
                all_transcripts_text.append(f"{video_url} [{start_time} - {end_time}] {text}")
                transcript_lines.append(text)
                timestamps.append((start_time, end_time))
                video_urls.append(video_url)


    # Encode transcript lines using the tokenizer
    transcript_inputs = tokenizer(transcript_lines, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Calculate cosine similarities between query and transcript embeddings
    with torch.no_grad():
        query_embedding = model.encoder(inputs).last_hidden_state.mean(dim=1)  # Query embedding
        transcript_embeddings = model.encoder(transcript_inputs['input_ids']).last_hidden_state.mean(dim=1)

    cosine_similarities = cosine_similarity(query_embedding.numpy(), transcript_embeddings.numpy())

    relevant_sections = []

    # Gather all transcript snippets that exceed the threshold similarity
    relevant_snippets = []
    for idx, score in enumerate(cosine_similarities[0]):
        if score > threshold:
            relevant_snippets.append((video_urls[idx], transcript_lines[idx], timestamps[idx]))  # Store text with its timestamps

    # Sort the relevant snippets by timestamps to ensure correct order
    relevant_snippets.sort(key=lambda x: x[2][0])

    # Format the relevant snippets and align with their timestamps in the desired output format
    formatted_relevant_snippets = []
    for (video_url, snippet, (start_time, end_time)) in relevant_snippets:
        formatted_relevant_snippets.append(
            f"Video:{video_url}\n[{start_time:.2f}s - {end_time:.2f}s] {snippet}")

    # Return the formatted relevant sections
    return [{"transcript": snippet, "score": 1.0} for snippet in formatted_relevant_snippets]


# Simulating your process functions for this demonstration
def process_transcripts(input_urls, progress_bar, status_text):
    total_steps = 100  # Example total steps for the process
    start_time = time.time()  # Track the start time
    for step in range(total_steps):
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        time_remaining = elapsed_time / (step + 1) * (total_steps - step - 1)  # Estimate remaining time
        time_remaining_str = f"{time_remaining:.2f} seconds remaining"  # Format remaining time

        time.sleep(0.1)  # Simulate a task
        progress_bar.progress(step + 1, text=f"Extracting transcripts: {step + 1}% done")
        status_text.text(time_remaining_str)  # Update the remaining time text

    return "Transcripts Extracted!"  # Once complete


def extract_timestamps_from_section(section):
    try:
        # Strip any leading/trailing whitespaces
        section = section.strip()

        # Check if the section contains timestamp information in the correct format
        if '[' not in section or ']' not in section:
            return None  # Skip sections that do not contain timestamps in '[start_time - end_time]' format

        # Extract the timestamp part of the section (the part inside the brackets)
        timestamp_part = section[section.find('[') + 1:section.find(']')].strip()  # Extract content inside brackets
        times = timestamp_part.split(" - ")

        # Ensure two timestamps are found in the section
        if len(times) != 2:
            return None  # Return None to skip this section

        # Clean timestamps and remove any unnecessary decimal precision
        start_time = float(times[0].strip().replace("s", ""))
        end_time = float(times[1].strip().replace("s", ""))

        # Round to a reasonable precision (e.g., 2 decimal places)
        start_time = round(start_time, 2)
        end_time = round(end_time, 2)

        return start_time, end_time
    except Exception as e:
        print(f"Error extracting timestamps from section '{section}'. Exception: {e}")
        return None  # Return None in case of an error


def extract_video_segments(input_string):
    # This pattern looks for YouTube URLs with timestamps in the format of [start_time - end_time]
    pattern = r"(https://www\.youtube\.com/watch\?v=[\w-]+(?:&t=\d+s)?)\s*\[([\d\.]+s)\s*-\s*([\d\.]+s)\]"

    # Find all matching segments
    matches = re.findall(pattern, input_string)

    video_segments = []
    last_end_time = 0

    # For each match, process the video URL and timestamps
    for match in matches:
        url, start, end = match
        start_time = float(start[:-1])  # Remove the 's' and convert to float
        end_time = float(end[:-1])  # Remove the 's' and convert to float
        # Ensure that the current segment does not overlap with the previous one
        if start_time < last_end_time:
            start_time = last_end_time  # Update the start time to avoid overlap

        # Update last_end_time to the end of the current segment
        last_end_time = end_time
        video_segments.append((url, start_time, end_time))

    return video_segments


import moviepy.editor as mp
from moviepy.editor import VideoFileClip

def clip_and_merge_videos(segments, downloaded_video_path, output_filename):
    temp_dir = "temp_vid"  # Name of the temporary directory
    total_duration = 0

    # Create the directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Create the directory

    # Full output path for the final video
    output_path = os.path.join(temp_dir, output_filename)
    temp_clips = []

    for segment in segments:
        # Extract the video URL, start and end times
        url, start_time, end_time = segment

        if not downloaded_video_path or not isinstance(downloaded_video_path, str):
            raise ValueError("Invalid video path provided")

        # Ensure the file exists before proceeding
        if not os.path.exists(downloaded_video_path):
            raise FileNotFoundError(f"Video file not found at: {downloaded_video_path}")

       # Using moviepy to clip both audio and video
        video_clip = mp.VideoFileClip(downloaded_video_path)
        # Ensure that the end_time does not exceed the video's duration
        end_time = min(end_time, video_clip.duration)
        video_clip = video_clip.subclip(start_time, end_time)
        min_clip_duration = 1.0
        # Check the duration of the clip before adding it to the list
        clip_duration = video_clip.duration
        if clip_duration < min_clip_duration:
            st.warning(
                f"Skipping clip with duration {clip_duration} seconds, as it's below the minimum threshold of {min_clip_duration} seconds.")
            video_clip.close()
            continue  # Skip this clip if its duration is too small

        temp_clips.append(video_clip)  # Add the video clip to the list
        clip_duration = video_clip.duration
        total_duration += clip_duration  # Add the clip duration to the total


    # Convert total duration to minutes
    total_duration_minutes = total_duration / 60
    st.text(f"Total duration of combined clips: {total_duration_minutes} minutes")

    # Combine all clips into a final video (both video and audio)
    if temp_clips:
        final_clip = mp.concatenate_videoclips(temp_clips)

        # Write the final video with audio
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)

        # Clean up temporary clips
        for clip in temp_clips:
            clip.close()
        # Clean up the downloaded video after merging
        if os.path.exists(downloaded_video_path):
            os.remove(downloaded_video_path)

        return output_path  # Return the path to the merged video
    else:
        st.text("No clips to merge")
        return "No clips to merge"



def main():
    st.set_page_config(page_title="Video & Playlist Processor", page_icon="🎬", layout="wide")

    st.markdown("""
    <style>
        .css-1d391kg {padding: 30px;}
        .stTextArea>div>div>textarea {
            font-size: 14px;
            line-height: 1.8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #ff5c5c;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #ff7d7d;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Video and Playlist Processor")

    input_urls = st.text_input("Enter YouTube Playlist(s) or Video URL(s) or both (comma-separated): \n\n Example of link: https://www.youtube.com/watch?v=abc123xyz or https://www.youtube.com/playlist?list=xyz456abc")

    if 'stored_transcripts' not in st.session_state:
        st.session_state.stored_transcripts = []
    if 'transcript_text' not in st.session_state:
        st.session_state.transcript_text = ""

    # Always display the buttons, not dependent on the input
    col1, col2 = st.columns([3, 1])

    with col1:
        extract_button = st.button("Extract Transcripts")

    if extract_button:
        progress_bar = col2.progress(0, text="Starting transcript extraction Please Hold...")
        status_text = col2.empty()  # Placeholder for dynamic status updates

        st.session_state.stored_transcripts = process_input(input_urls)
        progress_bar.progress(50, text="Processing transcripts...")
        status_text.text("Processing transcripts...")
        progress_bar.progress(100, text="Transcripts extracted successfully.")
        status_text.text("Transcripts extracted successfully.")
        if st.session_state.stored_transcripts:
            transcript_text = ""
            for video in st.session_state.stored_transcripts:
                transcript_text += f"\nTranscript for video {video['video_url']}:\n"
                if isinstance(video['transcript'], list):
                    for line in video['transcript']:
                        transcript_text += line + "\n"
                else:
                    transcript_text += video['transcript'] + "\n"
                transcript_text += "-" * 50 + "\n"
            st.session_state.transcript_text = transcript_text

    if st.session_state.transcript_text:
        st.subheader("Extracted Transcripts")
        st.text_area("Transcripts", st.session_state.transcript_text, height=300, key="transcripts_area")

    query = st.text_input("Enter your query to extract relevant information:")

    # Now the "Process Query" button will be below the query input field
    process_query_button = st.button("Process Query")  # Always visible button

    if process_query_button and query:
        progress_bar = col2.progress(0, text="Starting query processing...")
        status_text = col2.empty()

        relevant_sections = process_query(query, st.session_state.stored_transcripts)
        progress_bar.progress(50, text="Analyzing query...")
        status_text.text("Analyzing query...")
        progress_bar.progress(100, text="Query processed successfully.")
        status_text.text("Query processed successfully.")
        if relevant_sections:
            st.session_state.query_output = "\n".join([section['transcript'] for section in relevant_sections])
        else:
            st.session_state.query_output = "No relevant content found for the query."

    if 'query_output' in st.session_state and st.session_state.query_output:
        st.subheader("Relevant Output for Your Query")
        st.text_area("Query Output", st.session_state.query_output, height=300, key="query_output_area")

    # File uploader for cookie file (either .json or .txt)
    cookie_file = st.file_uploader(
        "Upload a Cookie File (JSON or TXT format)\n\n You can find and upload the cookie file by following these steps: \n\n1. Open Developer Tools (F12 or right-click > Inspect).\n2. Go to the 'Application' tab and select 'Cookies'.\n3. Export cookies using either 'Copy All as HAR' or the EditThisCookie extension (https://chrome.google.com/webstore/detail/editthiscookie).\n4. Save as .json or .txt file and upload below.",
        type=["json", "txt"]
    )

    # Convert cookies to Netscape format if a file is uploaded
    netscape_cookies = None
    if cookie_file:
        netscape_cookies = convert_cookies_to_netscape(cookie_file)
        if netscape_cookies:
            cookie_path = "cookies_netscape.txt"
            with open(cookie_path, "w") as f:
                f.write("\n".join(netscape_cookies))  # Save as a Netscape cookie file
            st.success(f"Cookie file converted to Netscape format and saved as {cookie_path}")

    if st.button("Combine and Play"):
        if 'query_output' in st.session_state and st.session_state.query_output:
            downloaded_video_paths = []
            # Ensure that `input_urls` is set and split correctly
            if input_urls:
                for url in input_urls.split(","):
                    url = url.strip()  # Clean the URL
                    # Call the download_video function to download the video and get the path
                    download_status, downloaded_video_path = download_video(url)  # Get the download status and path
                    if downloaded_video_path:  # Check if a valid path was returned
                        downloaded_video_paths.append(downloaded_video_path)  # Add the path to the list
            video_segments = extract_video_segments(st.session_state.query_output)
            output_filename = "final_video.mp4"
            final_path = clip_and_merge_videos(video_segments, downloaded_video_path, output_filename)
            # Check if the final video file exists
            if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                st.success("Final video created successfully!")
                st.video(final_path)  # Display the final video
            else:
                st.error("Failed to create the final video. Please check the video segments and try again.")
        else:
            st.error("No segments to combine. Process a query first.")

if __name__ == "__main__":
    main()
