import toml
#import youtube_dl
from pytube import YouTube
import os
import assemblyai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import openai
import hashlib

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import requests
import json
import time
from datetime import datetime

app_config={}

#
# AssemblyAI integration. Code directly from their website.
#

def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data

def upload_file(api_token, path):
    headers = {'authorization': api_token}
    response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=read_file(path))

    if response.status_code == 200:
        return response.json()["upload_url"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def create_transcript(api_token, audio_url):
    url = "https://api.assemblyai.com/v2/transcript"
    headers = { "authorization": api_token, "content-type": "application/json" }
    data = { "audio_url": audio_url, "speaker_labels": True, "speakers_expected": 3 }

    response = requests.post(url, json=data, headers=headers)
    transcript_id = response.json()['id']
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()
        if transcription_result['status'] == 'completed':
            break
        elif transcription_result['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
        else:
            time.sleep(3)
    return transcription_result

#
# Credentials. This should be done vis ENV variables.
#
def setEnvVariables():
  global app_config
  try:
    with open(".streamlit/secrets.toml", "r") as f:
      app_config = toml.load(f)
  except FileNotFoundError: # Assume we are in streamlit cloud
    for k,v in st.secrets:
      app_config[k]=v
  print(f"App Config is {app_config}")

#
# Helper methods
#
def transcribeVideo(video_url='https://www.youtube.com/watch?v=kMPBcXdSwiY'):
  yt = YouTube(video_url)
  audio = yt.streams.filter(only_audio=True).first()
  out_file = audio.download(output_path=".")

  base, ext = os.path.splitext(out_file)
  new_file = base + '.mp3'
  json_file = base + '.json'
  os.rename(out_file, new_file)

  assemblyai_api_token=app_config['ASSEMBLYAI_API_TOKEN']
  assemblyai_upload_url = upload_file(assemblyai_api_token, new_file)
  transcript = create_transcript(assemblyai_api_token, assemblyai_upload_url)
  metadata={"filename":new_file,"video_url":video_url}
  print(f'{datetime.now()} Completed Transcribing video:{metadata}')
  with open(json_file,"w") as f:
    json.dump(transcript["utterances"],f,indent=4)
  return (metadata,transcript)

import requests

def upload_to_pinecone(text,embeddings,metadata,pineconeNamespace,pineconeIndex,pineconeKey):
  index = pinecone.Index(pineconeIndex)
  r=index.upsert([(text,embeddings,metadata)],namespace=pineconeNamespace)
  return r

def delete_namespace(pineconeIndex,pineconeNamespace):
  index = pinecone.Index(pineconeIndex)
  index.delete(deleteAll='true', namespace=pineconeNamespace)


def uploadEmbeddings(transcript,metadata):
  openai.api_key=app_config['OPENAI_API_KEY']
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
  pinecone.init( api_key=app_config['PINECONE_API_KEY'], environment=app_config['PINECONE_API_ENV'])

  # delete_namespace(PINECONE_INDEX_NAME,PINECONE_NAMESPACE)

  uts=transcript["utterances"]
  for ut in uts:
    t=ut["text"]
    sp=ut["speaker"]
    st=ut["start"]
    en=ut["end"]
  
    texts = text_splitter.create_documents([t])
    for text in texts:
      embedding=openai.Embedding.create(model="text-embedding-ada-002",input=text.page_content)
      query_result=embedding['data'][0]['embedding']
      metadata2={"starttime":st,"endtime":en,"speaker":sp,"text":text.page_content}|metadata
      #print(len(text.page_content),text.page_content,metadata2,query_result)
      hash = hashlib.md5(text.page_content.encode("utf-8"))
      print(hash.hexdigest())
      r=upload_to_pinecone(hash.hexdigest(),query_result,metadata2,app_config['PINECONE_NAMESPACE'], app_config['PINECONE_INDEX_NAME'],app_config['PINECONE_API_KEY'])
      print(f'{datetime.now()} Pinecone uploaded:{len(text.page_content)}')

def processVideo(video_url='https://www.youtube.com/watch?v=kMPBcXdSwiY'):
  (m,t)=transcribeVideo(video_url)
  uploadEmbeddings(t,m)
#
#
#
setEnvVariables()
print("Hello World")
#processVideo('https://www.youtube.com/watch?v=T3JLARdS748')
#processVideo('https://www.youtube.com/watch?v=hguu1hxqnhY')
#processVideo('')
processVideo('https://www.youtube.com/watch?v=8nxWpfbnNQ0')
print("Bye World")
