import toml
import streamlit as st
import os
import pinecone
import openai
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from platform import python_version

#
# Credentials. This should be done vis ENV variables.
#
app_config={}
def setEnvVariables():
  global app_config
  try:
    with open(".streamlit/secrets.toml", "r") as f:
      app_config = toml.load(f)
  except FileNotFoundError: # Assume we are in streamlit cloud
    for k,v in st.secrets:
      app_config[k]=v
  print(f"App Config is {app_config}")

def getMatchingDocs(q):
  openai.api_key=app_config['OPENAI_API_KEY'] 
  pinecone.init( api_key=app_config['PINECONE_API_KEY'], environment=app_config['PINECONE_API_ENV'])
  model='text-embedding-ada-002'
  embed_query=openai.Embedding.create(input=q,engine=model)
  query_embeds=embed_query['data'][0]['embedding']
  #st.dataframe(query_embeds)
  index=pinecone.Index(app_config['PINECONE_INDEX_NAME'])
  #print(query_embeds)
  response=index.query(query_embeds,top_k=3,include_metadata=True,namespace=app_config['PINECONE_NAMESPACE'])
  #print(response['matches'])
  li=[]
  for r in response['matches']:
    rmd=r["metadata"]
    sc=r.get('score','')
    tx=rmd.get('text','')
    fn=rmd.get('filename','')
    vu=rmd.get('video_url','')
    stt=rmd.get('starttime','')
    et=rmd.get('endtime','')
    li.append({"score":sc,"text":tx,"filename":fn,"video_url":vu,"start":stt,"end":et})
  print(f"List is {li}")
  st.dataframe(li,use_container_width=True)
  return li

def youtubeVideoOffset(url,offset):
  if('UNKNOWN' in url):
    return url
  if('-1' in offset):
    return url
  os='t='+str(int(float(offset)/1000))+'s'
  if('?' in url):
    return url+'&'+os
  else:
    return url+'?'+os


def runQuery(q,li):
  llm = OpenAI(temperature=0, openai_api_key=app_config['OPENAI_API_KEY'])
  chain = load_qa_with_sources_chain(llm, chain_type="stuff")
  docs=[]
  for l in li:
    pc=l.get('text','')
    l.pop('text')
    yt_url=l.get('video_url',"UNKNOWN")
    yt_ts=str(l.get('start','-1'))
    #yt_url=yt_url+'?t='+str(int(yt_ts))+'s'
    l["source"]=youtubeVideoOffset(yt_url,yt_ts)
    docs.append(Document(page_content=pc,metadata=l))
  s=chain.run(input_documents=docs, question=q)
  print(f"\nChain Run returned: {s}")
  st.write(s)

setEnvVariables()
text_input=st.text_input("Enter your query:")
if st.button('Submit'):
  m=getMatchingDocs(text_input)
  s=runQuery(text_input,m)
