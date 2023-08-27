import pandas as pd
from yt_stats import YTstats
from tqdm import tqdm
import numpy as np

API_KEY = 'AIzaSyCarGpEfvHg-9zsYEMpbZ754o1OvsuBvok'

df = pd.DataFrame(np.array([["M&S Christmas�s TV Ad from 2017", "KfaSxIkLslE","marksandspencertv"]]), columns=["Title", "Video ID", "Channel ID"])



Views =[]

for x,y in tqdm(zip(df["Video ID"],df["Channel ID"])):
    channel_id = y
    video_id = x
    part = 'statistics'
    yt = YTstats(API_KEY, channel_id)
    a = yt._get_single_video_data(video_id,part)
    Views.append(a['viewCount'])

df["views"] = Views


print(df["views"])

