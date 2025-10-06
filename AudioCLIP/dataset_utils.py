import os
import pathlib
from pathlib import Path
from os import PathLike
import subprocess

from pytube import YouTube, Search
from yt_dlp import YoutubeDL
from typing import List, Tuple
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic_models import DatasetCreationResponse
from pydantic import SecretStr

load_dotenv(find_dotenv())

MODEL = 'x-ai/grok-4-fast:free'
API_KEY = SecretStr(os.environ.get('OPENROUTER_API_KEY'))
BASE_URL = os.environ.get('OPENROUTER_BASE_URL')
assert API_KEY is not None, "Please set the OPENROUTER_API_KEY environment variable."

llm = ChatOpenAI(api_key=API_KEY, model=MODEL, base_url=BASE_URL, )


def generate_classes(count: int, video_query_multiplier: int = 5) -> DatasetCreationResponse:
    system_prompt = "You are a helpful assistant that generates a list of classes and video queries to use in a Video Retrieval Task demo."
    prompt = """Please generate the response according to the following JSON schema:
{response_schema}
The total count of classes should be {count}.
The total count of video queries should be {query_count}.
Follow these rules:
- Each class should be a short noun phrase, ideally 1-3 words.
- Each class should be unique and not overlap with other classes.
- Each class should have multiple video queries (to search youtube for videos) associated with it.
- The queries will be used to search youtube for videos, so they should be realistic search queries.
- Note that this demo is for video retrieval. You may include irrelevant classes to show the model's ability to distinguish between relevant and irrelevant videos."""
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", prompt)])
    chain = prompt_template | llm.with_structured_output(DatasetCreationResponse.model_json_schema()).with_retry()
    response = chain.invoke({"count": count, "query_count": count * video_query_multiplier,
                             "response_schema": DatasetCreationResponse.model_json_schema()})

    return response


def search_yt(query: str, count=5) -> List[str]:
    search = Search(query)
    video_urls = []
    for result in search.results[:count]:
        video_urls.append(result.watch_url)
    return video_urls


def download_video(url: str, output_filename: PathLike, extension: str = 'mkv') -> Path:
    ydl_opts = {'format': 'ba+bv', 'N': 8, 'outtmpl': f"{output_filename}.{extension}",
                'merge_output_format': extension, 'quiet': True, 'no_warnings': True,
                'extractor_args': {"youtube": {"formats": "dashy"}}}
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return pathlib.Path(f"{output_filename}.{extension}")


def process_audio(video_path: PathLike, output_path: PathLike, name_prefix: str, sample_rate: int = 44100,
                  mono: bool = True, codec: str = 'pcm_s16le', container: str = 'wav', cut_interval: int = 5) -> None:
    """
    Extract audio from video and split into segments with start timecode in filename.
    Filename format: prefix_STARTTIMESTAMP.ext (e.g., video_0.000.wav, video_5.000.wav)
    """
    input_video = Path(video_path)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video file {input_video} does not exist.")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_output_template = output_path / f"{name_prefix}_%d.{container}"

    ffmpeg_command = ['ffmpeg', '-i', str(input_video), '-vn',  # No video
                      '-acodec', codec, '-ar', str(sample_rate), '-ac', '1' if mono else '2', '-f', 'segment',
                      '-segment_time', str(cut_interval), '-segment_time_delta', '0.01',  # More precise segment timing
                      '-reset_timestamps', '1', '-frame_pts', '1',  # Use PTS in filename
                      str(audio_output_template)]

    try:
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        raise


def process_frames(video_path: PathLike, output_path: PathLike, name_prefix: str, frame_sample_interval: int = 5,
                   extension: str = 'jpg') -> None:
    """
    Extract frames from video at specified intervals with timestamp in filename.
    Filename format: prefix_TIMESTAMP.ext (e.g., video_0.jpg, video_10.jpg, video_20.jpg)
    """
    input_video = Path(video_path)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video file {input_video} does not exist.")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    frames_output_template = output_path / f"{name_prefix}_%d.{extension}"

    ffmpeg_command = ['ffmpeg', '-i', str(input_video), '-vf', f'fps=1/{frame_sample_interval}', '-frame_pts', '1',
                      # Use PTS (timestamp) in filename
                      '-vsync', '0',  # Don't duplicate/drop frames
                      str(frames_output_template)]

    try:
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        raise


def process_downloaded_video(video_path: PathLike, output_dir: PathLike, interval: int = 5) -> Tuple[Path, Path]:
    input_video = Path(video_path)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video file {input_video} does not exist.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_out_dir = output_dir / 'audio'
    frames_out_dir = output_dir / 'frames'

    name_prefix = input_video.stem
    process_audio(input_video, audio_out_dir, name_prefix, cut_interval=interval)
    process_frames(input_video, frames_out_dir, name_prefix, frame_sample_interval=interval)

    return audio_out_dir, frames_out_dir
