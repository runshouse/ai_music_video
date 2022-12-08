import flask
from flask import Flask, render_template # for web app
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

#Import libraries
from omegaconf import OmegaConf, DictConfig
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from pathlib import Path
# import time

# from vktrs.utils import sanitize_folder_name

# import copy
# import datetime as dt
# import gc
# from itertools import chain, cycle
# import json
# import os
# import re
# import string
# from subprocess import Popen, PIPE
# import textwrap
# import time
# import warnings

# from IPython.display import display
# import numpy as np
# import pandas as pd
# import panel as pn
# from tqdm.autonotebook import tqdm

# import tokenizations
# import webvtt
# import whisper

# from vktrs.utils import remove_punctuation
# from vktrs.utils import get_audio_duration_seconds
# from vktrs.youtube import (
#     YoutubeHelper,
#     parse_timestamp,
#     vtt_to_token_timestamps,
#     srv2_to_token_timestamps,
# )


# #Configure Notebook
# cfg = OmegaConf.create({
#     'active_project':str(time.time()),
#     'project_root':proj_root_str,
#     'gdrive_mounted':mount_gdrive,
#     'use_stability_api':use_stability_api,
#     'model_dir':model_dir_str,
#     'output_dir':'${active_project}/frames'
# })

# with open('config.yaml','w') as fp:
#     OmegaConf.save(config=cfg, f=fp.name)

# #Create Folder
# project_name = 'Project Folder'
# if not project_name:
#     project_name = str(time.time())

# project_name = sanitize_folder_name(project_name)

# workspace = OmegaConf.load('config.yaml')
# workspace.active_project = project_name

# with open('config.yaml','w') as fp:
#     OmegaConf.save(config=workspace, f=fp.name)

# #Reset workspace
# if 'df' in locals():
#     del df
# if 'df_regen' in locals():
#     del df_regen

#############################################
#Function to Get Lyrics from Youtube Video
#############################################
# def get_lyrics(youtube_url):
#     url = youtube_url
#     workspace = OmegaConf.load('config.yaml')

#     model_dir = workspace.model_dir

#     root = workspace.project_root
#     root = Path(root)
#     root.mkdir(parents=True, exist_ok=True)

#     storyboard = OmegaConf.create()

#     d_ = dict(
#         # all this does is make it so each of the following lines can be preceded with a comma
#         # otw the first parameter would be offset from the other in the colab form
#         _=""

#         , video_url = youtube_url
#         , audio_fpath = '' # @param {type:'string'}
#         , whisper_seg = True # @param {type:'boolean'}
#     )
#     d_.pop('_')
#     storyboard.params = d_

#     if not storyboard.params.audio_fpath:
#         storyboard.params.audio_fpath = None

#     storyboard_fname = root / 'storyboard.yaml'
#     with open(storyboard_fname,'wb') as fp:
#         OmegaConf.save(config=storyboard, f=fp.name)

#         # Download audio from youtube #
#         video_url = storyboard.params.video_url

#         if video_url:
#             # check if user provided an audio filepath (or we already have one from youtube) before attempting to download
#             if storyboard.params.get('audio_fpath') is None:
#                 helper = YoutubeHelper(
#                     video_url,
#                     ydl_opts = {
#                         'outtmpl':{'default':str( root / f"ytdlp_content.%(ext)s" )},
#                         'writeautomaticsub':True,
#                         'subtitlesformat':'srv2/vtt'
#                         },
#                 )
#                 # estimate video end
#                 video_duration = dt.timedelta(seconds=helper.info['duration'])
#                 storyboard.params['video_duration'] = video_duration.total_seconds()

#                 audio_fpath = str( root / 'audio.mp3' )
#                 input_audio = helper.info['requested_downloads'][-1]['filepath']
#                 # !ffmpeg -y -i "{input_audio}" -acodec libmp3lame {audio_fpath}

#                 # to do: write audio and subtitle paths/meta to storyboard
#                 storyboard.params.audio_fpath = audio_fpath

#                 if False:
#                     subtitle_format = helper.info['requested_subtitles']['en']['ext']
#                     subtitle_fpath = helper.info['requested_subtitles']['en']['filepath']

#                     if subtitle_format == 'srv2':
#                         with open(subtitle_fpath, 'r') as f:
#                             srv2_xml = f.read()
#                         token_start_times = srv2_to_token_timestamps(srv2_xml)
#                         # to do: handle timedeltas...
#                         #storyboard.params.token_start_times = token_start_times

#                     elif subtitle_format == 'vtt':
#                         captions = webvtt.read(subtitle_fpath)
#                         token_start_times = vtt_to_token_timestamps(captions)
#                         # to do: handle timedeltas...
#                         #storyboard.params.token_start_times = token_start_times

#                     # If unable to download supported subtitles, force use whisper
#                     else:
#                         storyboard.params.whisper_seg = True


#     # estimate video end
#     if storyboard.params.get('video_duration') is None:
#         # estimate duration from audio file
#         audio_fpath = storyboard.params['audio_fpath']
#         storyboard.params['video_duration'] = get_audio_duration_seconds(audio_fpath)

#     if storyboard.params.get('video_duration') is None:
#         raise RuntimeError('unable to determine audio duration. was a video url or path to a file supplied?')

#     # force use
#     storyboard.params.whisper_seg = True

#     with open(storyboard_fname,'wb') as fp:
#         OmegaConf.save(config=storyboard, f=fp.name)

#     whisper_seg = storyboard.params.whisper_seg

#     # 💬 Transcribe and segment speech using whisper #
#     # handle OOM... or try to, anyway
#     if 'hf_helper' in locals():
#         del hf_helper.img2img
#         del hf_helper.text2img
#         del hf_helper

#     if whisper_seg:
#         from vktrs.asr import (
#             #whisper_lyrics,
#             #whisper_transcribe,
#             #whisper_align,
#             whisper_transmit_meta_across_alignment,
#             whisper_segment_transcription,
#         )

#         #prompt_starts = whisper_lyrics(audio_fpath=storyboard.params.audio_fpath)

#         audio_fpath = storyboard.params.audio_fpath
#         #whispers = whisper_transcribe(audio_fpath)

#         # to do: dropdown selectors
#         segmentation_model = 'tiny'
#         transcription_model = 'large'

#         storyboard.params.whisper = dict(
#             segmentation_model = segmentation_model
#             ,transcription_model = transcription_model
#         )

#         whispers = {
#             #'tiny':None, # 5.83 s
#             #'large':None # 3.73 s
#         }

#         for k in set([segmentation_model, transcription_model]):
#             #if k in scripts:

#             options = whisper.DecodingOptions(
#                 language='en',
#             )
#             # to do: be more proactive about cleaning up these models when we're done with them
#             model = whisper.load_model(k).to('cuda')
#             start = time.time()
#             print(f"Transcribing audio with whisper-{k}")

#             # to do: calling transcribe like this unnecessarily re-processes audio each time.
#             whispers[k] = model.transcribe(audio_fpath) # re-processes audio each time, ~10s overhead?
#             print(f"elapsed: {time.time()-start}")
#             del model
#             gc.collect()

#         #######################
#         # save transcriptions #
#         #######################

#         transcriptions = {}
#         transcription_root = root / 'whispers'
#         transcription_root.mkdir(parents=True, exist_ok=True)
#         for k in whispers:
#             outpath = str( transcription_root / f"{k}.vtt" )
#             transcriptions[k] = outpath
#             with open(outpath,'w') as f:
#                 # to do: upstream PR to control verbosity
#                 whisper.utils.write_vtt(
#                     whispers[k]["segments"], # ...really?
#                     file=f
#                 )
#         storyboard.params.whisper.transcriptions = transcriptions

#         #tiny2large, large2tiny, whispers_tokens = whisper_align(whispers)
#         # sanitize and tokenize
#         whispers_tokens = {}
#         for k in whispers:
#             whispers_tokens[k] = [
#             remove_punctuation(tok) for tok in whispers[k]['text'].split()
#             ]

#         # align sequences
#         tiny2large, large2tiny = tokenizations.get_alignments(
#             whispers_tokens[segmentation_model], #whispers_tokens['tiny'],
#             whispers_tokens[transcription_model] #whispers_tokens['large']
#         )
#         #return tiny2large, large2tiny, whispers_tokens

#         token_large_index_segmentations = whisper_transmit_meta_across_alignment(
#             whispers,
#             large2tiny,
#             whispers_tokens,
#         )
#         prompt_starts = whisper_segment_transcription(
#             token_large_index_segmentations,
#         )

#         ### checkpoint the processing work we've done to this point

#         prompt_starts_copy = copy.deepcopy(prompt_starts)

#         # to do: deal with timedeltas in asr.py and yt.py
#         for rec in prompt_starts_copy:
#             for k,v in list(rec.items()):
#                 if isinstance(v, dt.timedelta):
#                     rec[k] = v.total_seconds()

#         storyboard.prompt_starts = prompt_starts_copy

#         with open(storyboard_fname) as fp:
#             OmegaConf.save(config=storyboard, f=fp.name)

#     pn.extension('tabulator') # I don't know that specifying 'tabulator' here is even necessary...

#     tabulator_formatters = {
#         'bool': {'type': 'tickCross'}
#     }

#     # reset workspace
#     if 'df_regen' in locals():
#         del df_regen

#     df = pd.DataFrame(prompt_starts).rename(
#         columns={
#             'ts':'Timestamp (sec)',
#             'prompt':'Lyric',
#         }
#     )

#     if 'td' in df:
#         del df['td']

#     df['override_prompt'] = ''

#     df_pre = copy.deepcopy(df)
#     pn.widgets.Tabulator(df, formatters=tabulator_formatters)
	# return render_template('sentiment.html', table=df.to_html(classes='data'))






def get_news(ticker):
    url = finviz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
    response = urlopen(req)
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    return news_table

# parse news into dataframe
def parse_news(news_table):
    parsed_news = []

    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text()
        # splite text in the td tag into a list
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]

        # else load 'date' as the 1st element and 'time' as the second
        else:
            date = date_scrape[0]
            time = date_scrape[1]

        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([date, time, text])

        # Set column names
        columns = ['date', 'time', 'headline']

        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
        parsed_news_df = pd.DataFrame(parsed_news, columns=columns)

        # Create a pandas datetime object from the strings in 'date' and 'time' column
        parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])

    return parsed_news_df

def score_news(parsed_news_df):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')


    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')

    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)

    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

    return parsed_and_scored_news

def plot_hourly_sentiment(parsed_and_scored_news, ticker):

    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('H').mean()

    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Hourly Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

def plot_daily_sentiment(parsed_and_scored_news, ticker):

    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('D').mean()

    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Daily Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment',methods = ['POST'])
def sentiment():

	ticker = flask.request.form['ticker'].upper()
	news_table = get_news(ticker)
	parsed_news_df = parse_news(news_table)
	parsed_and_scored_news = score_news(parsed_news_df)
	fig_hourly = plot_hourly_sentiment(parsed_and_scored_news, ticker)
	fig_daily = plot_daily_sentiment(parsed_and_scored_news, ticker)

	graphJSON_hourly = json.dumps(fig_hourly, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_daily = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)

	header= "Hourly and Daily Sentiment of {} Stock".format(ticker)
	description = """
	The above chart averages the sentiment scores of {} stock hourly and daily.
	The table below gives each of the most recent headlines of the stock and the negative, neutral, positive and an aggregated sentiment score.
	The news headlines are obtained from the FinViz website.
	Sentiments are given by the nltk.sentiment.vader Python library.
    """.format(ticker)
	return render_template('sentiment.html',graphJSON_hourly=graphJSON_hourly, graphJSON_daily=graphJSON_daily, header=header,table=parsed_and_scored_news.to_html(classes='data'),description=description)




if __name__ == '__main__':
    app.run()
