# -*- coding: utf-8 -*-
"""
Created on Mon May  9 03:09:56 2022

@author: sreek
"""

import pandas as pd
import requests
import streamlit as st 
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt


# cwd = os.chdir("C:\\Algo Trading\\Data")

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
            'accept-language': 'en,gu;q=0.9,hi;q=0.8',
            'accept-encoding': 'gzip, deflate, br'}

daterange = pd.bdate_range("03-12-2021", "03-11-2022")

nseFNOSecurities_df = pd.read_csv("C:\\Algo Trading\\Data\\SecurityMaster\\FONSEScripMaster.txt")[['InstrumentName', 'ExchangeCode']]
nseFNOSecurities_df = nseFNOSecurities_df[nseFNOSecurities_df['InstrumentName']=="FUTSTK"].\
                        drop_duplicates('ExchangeCode', keep='last').\
                            drop(['InstrumentName'],axis=1).\
                                sort_values(by=['ExchangeCode'])
                                
new_row = pd.DataFrame({"ExchangeCode":["NIFTY","BANKNIFTY"]})
nseFNOSecurities_df1 = pd.concat([new_row,nseFNOSecurities_df])
ce_trigger = pd.DataFrame(columns=['STOCK','LTP',"CE_STRIKE",'CE_DIFF','CE_OI','CE_OICHG'])
pe_trigger = pd.DataFrame(columns=['STOCK','LTP',"PE_STRIKE",'PE_DIFF','PE_OI','PE_OICHG'])

def get_session_cookie():
    
    url_oc      = "https://www.nseindia.com/option-chain"
    sess_Cookie = requests.Session()
    cookies = dict()

    request = sess_Cookie.get(url_oc, headers=headers, timeout=5)
    cookies = dict(request.cookies)

    return cookies

def option_data(security,sesscookie,session):
    global ce_trigger
    global pe_trigger
    security = security.replace('&', '%26')
    url = 'https://www.nseindia.com/api/option-chain-equities?symbol={0}'.format(security)
    r = session.get(url,headers=headers,timeout=10,cookies=sesscookie).json()

    ce_data = pd.DataFrame([data["CE"] for data in r['filtered']['data'] if "CE" in data])[['pchangeinOpenInterest','openInterest','strikePrice','underlying','underlyingValue']]
    ce_data = ce_data[ce_data.openInterest == ce_data.openInterest.max()]
    ce_data["callDiff"] = round((ce_data["strikePrice"] - ce_data["underlyingValue"])*100/ce_data["underlyingValue"],2)
    ce_data.columns = ['CE_OICHG','CE_OI',"CE_STRIKE", 'STOCK', 'LTP', 'CE_DIFF']
    if (-1.2 <= ce_data['CE_DIFF'].iloc[-1] <= 1.2):
        ce_trigger =  pd.concat([ce_trigger,ce_data[['STOCK','LTP',"CE_STRIKE",'CE_DIFF','CE_OI','CE_OICHG']]],ignore_index=True)

    pe_data = pd.DataFrame([data["PE"] for data in r['filtered']['data'] if "PE" in data])[['underlying','strikePrice','openInterest','pchangeinOpenInterest','underlyingValue']]
    pe_data = pe_data[pe_data.openInterest == pe_data.openInterest.max()]
    pe_data["putDiff"] = round((pe_data["strikePrice"] - pe_data["underlyingValue"])*100/pe_data["underlyingValue"],2)
    pe_data.columns = ['STOCK',"PE_STRIKE",'PE_OI','PE_OICHG','LTP','PE_DIFF']
    if (-1.2 <= pe_data['PE_DIFF'].iloc[-1] <= 1.2):
        pe_trigger =  pd.concat([pe_trigger,pe_data[['STOCK','LTP',"PE_STRIKE",'PE_DIFF','PE_OI','PE_OICHG']]],ignore_index=True)


def getTriggerData():
    sesscookie = get_session_cookie()
    session = requests.Session()
    
    # securityList = ["WIPRO","INFY"]
    securityList = nseFNOSecurities_df["ExchangeCode"].tolist()
            
    for security in securityList:
        # print(security)
        Thread(target=option_data, args=(security,sesscookie,session)).start()

    return ce_trigger, pe_trigger

def plot_data(row):
    
    plot7 = 0
    
    if row['CE_OICHG'] >= 0:
        plot1 = row["CE_OI"]-row["CE_OICHG"]
        plot2 = row["CE_OICHG"]
        plot3 = 0
    else:
        plot1 = row["CE_OI"]
        plot2 = 0
        plot3 = abs(row["CE_OICHG"])
    
    if row['PE_OICHG'] >= 0:
        plot4 = row["PE_OI"]-row["PE_OICHG"]
        plot5 = row["PE_OICHG"]
        plot6 = 0
    else:
        plot4 = row["PE_OI"]
        plot5 = 0
        plot6 = abs(row["PE_OICHG"])
    
    if row["LTP"] == row["STRIKE"]:
        plot7 = row["DIFF"]
    
    return plot1, plot2, plot3,plot4, plot5,plot6,plot7

def map_FutOIdata(row):
    FH_LONG = 0
    FH_LONGU = 0
    FH_SHORT= 0 
    FH_SHORTC = 0
    if row['FH_CHANGE_IN_PRICE'] > 0:               #increase in price
        if row['FH_CHANGE_IN_OI'] > 0:              #Build up in OI - Long Build up 
            FH_LONG = row['FH_OPEN_INT']
        else:                                       #Unwinding in OI - Short Covering
            FH_SHORTC = row['FH_OPEN_INT']
    else:                                           #Decrease in price
        if row['FH_CHANGE_IN_OI'] > 0:              #Build up in OI - Short Build up 
            FH_SHORT = row['FH_OPEN_INT']
        else:                                       #Unwinding in OI - Long Unwinding
            FH_LONGU = row['FH_OPEN_INT']
    return FH_LONG, FH_LONGU, FH_SHORT, FH_SHORTC

def option_chain(security):
    
    if (security == 'NIFTY' or security == 'BANKNIFTY'): # set url to Index and set chain length to 10
        url = url = 'https://www.nseindia.com/api/option-chain-indices?symbol={0}'.format(security)
        chain = 8
    else: # set url to stock and set chain length to 10
        url = url = 'https://www.nseindia.com/api/option-chain-equities?symbol={0}'.format(security)
        chain = 5
        
    sesscookie = get_session_cookie()
    session = requests.Session()
    security = security.replace('&', '%26')
    
    r = session.get(url,headers=headers,timeout=10,cookies=sesscookie).json()

    ce_data = pd.DataFrame([data["CE"] for data in r['filtered']['data'] if "CE" in data])[['changeinOpenInterest','openInterest','strikePrice','underlying','underlyingValue']]
    ce_data["callDiff"] = abs(round((ce_data["strikePrice"] - ce_data["underlyingValue"])*100/ce_data["underlyingValue"],2))
    ce_data.columns = ['CE_OICHG','CE_OI',"STRIKE", 'STOCK', 'LTP', 'DIFF']

    pe_data = pd.DataFrame([data["PE"] for data in r['filtered']['data'] if "PE" in data])[['underlying','strikePrice','openInterest','changeinOpenInterest','underlyingValue']]
    pe_data["putDiff"] = abs(round((pe_data["strikePrice"] - pe_data["underlyingValue"])*100/pe_data["underlyingValue"],2))
    pe_data.columns = ['STOCK',"STRIKE",'PE_OI','PE_OICHG','LTP','DIFF']
    
    option_data = ce_data.merge(pe_data, how='inner', on = ['STRIKE','STOCK','DIFF','LTP'])

    idx = option_data[['DIFF']].idxmin().item()
    option_data = option_data.iloc[idx - chain : idx + chain+1]
    
    LTP = option_data["LTP"].max()
    maxOI =  max(option_data["CE_OI"].max(),option_data["PE_OI"].max())
    option_data.loc[len(option_data.index)] = [0,0,LTP, security,LTP,maxOI,0,0] 
    
    option_data['CE_PLOT1'], option_data['CE_PLOT2'],option_data['CE_PLOT3'],option_data['PE_PLOT1'], option_data['PE_PLOT2'],option_data['PE_PLOT3'], option_data['LTP_PLOT']= zip(*option_data.apply(plot_data, axis=1))
    option_data = option_data.sort_values(by=['STRIKE'])
    return option_data

def open_interest(security,duration):

    i = 1
    calDuration = duration*7/5 # in Calender days
    iteration = round((calDuration/21)+0.5)
    toDate = dt.datetime.today()
    sesscookie = get_session_cookie()
    session = requests.Session()
    security = security.replace('&', '%26')
    futOI = pd.DataFrame(columns=['FH_TIMESTAMP','FH_SYMBOL', 'FH_EXPIRY_DT','FH_PREV_CLS', 'FH_SETTLE_PRICE', 'FH_OPEN_INT','FH_CHANGE_IN_OI'])

    for i in range(iteration):
        if calDuration <= 21:
            fromDate = toDate - dt.timedelta(calDuration-1)
        else:
            fromDate = toDate - dt.timedelta(20)
        
        fromDateStr = fromDate.strftime("%d-%m-%Y")
        toDateStr = toDate.strftime("%d-%m-%Y")
        print(fromDateStr," - ", toDateStr)
        
        url = 'https://www.nseindia.com/api/historical/fo/derivatives?&from={}&to={}&instrumentType=FUTSTK&symbol={}'.format(fromDateStr,toDateStr,security)
        r = session.get(url,headers=headers,timeout=10,cookies=sesscookie).json()
        futOI =  pd.concat([futOI,pd.DataFrame(r["data"])[['FH_TIMESTAMP','FH_SYMBOL', 'FH_EXPIRY_DT','FH_PREV_CLS', 'FH_SETTLE_PRICE', 'FH_OPEN_INT','FH_CHANGE_IN_OI']]],ignore_index=True)
        toDate = fromDate - dt.timedelta(1)
        calDuration = calDuration - 21
        i = i+1

    futOI[["FH_PREV_CLS","FH_SETTLE_PRICE","FH_OPEN_INT", "FH_CHANGE_IN_OI"]] = futOI[["FH_PREV_CLS","FH_SETTLE_PRICE","FH_OPEN_INT", "FH_CHANGE_IN_OI"]].astype('double').astype('int')
    futOI["FH_TIMESTAMP"] = pd.to_datetime(futOI['FH_TIMESTAMP'])
    futOI = futOI.groupby(['FH_TIMESTAMP','FH_SYMBOL']).agg({'FH_PREV_CLS':'first','FH_SETTLE_PRICE':'first','FH_OPEN_INT':'sum','FH_CHANGE_IN_OI':'sum'}).reset_index().sort_values(by=['FH_TIMESTAMP'])
    futOI["FH_CHANGE_IN_PRICE"] = futOI["FH_SETTLE_PRICE"] -futOI["FH_PREV_CLS"]
    futOI["FH_LONG"], futOI["FH_LONGU"], futOI["FH_SHORT"], futOI["FH_SHORTC"] = zip(*futOI.apply(map_FutOIdata, axis=1))
    
    return futOI
    

def get_option_plot(df):
    
    strike = df["STRIKE"]
    
    x = np.arange(len(strike))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    ceplot1 = ax.barh(x + width/2, df["CE_PLOT1"], width, color='red', alpha= 0.6)
    ceplot2 = ax.barh(x + width/2, df["CE_PLOT2"], width, color='red', alpha= 1, left =df["CE_PLOT1"])
    ceplot3 = ax.barh(x + width/2, df["CE_PLOT3"], width, color='red', alpha= 0.2, left =(df["CE_PLOT1"]+df["CE_PLOT2"]))
    ltpplot = ax.barh(x + width/2, df["LTP_PLOT"], 0.1, color='blue', alpha= 1,linestyle='--')
    peplot1 = ax.barh(x - width/2, df["PE_PLOT1"], width, color='green', alpha=0.6)
    peplot2 = ax.barh(x - width/2, df["PE_PLOT2"], width, color='green', alpha=1, left =df["PE_PLOT1"])
    peplot3 = ax.barh(x - width/2, df["PE_PLOT3"], width, color='green', alpha=0.2, left =(df["PE_PLOT1"]+df["PE_PLOT2"]))
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Strike')
    ax.set_xlabel('Open Interest')
    
    #ax.set_title('Option Chain')
    ax.set_yticks(x, strike)
    # ax.legend()
    
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
       
    fig.tight_layout()
    
    return fig

def get_OI_plot(futOI):
    
    date = futOI["FH_TIMESTAMP"].dt.date
    x = np.arange(len(date))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax1 = plt.subplots()

    plt.ylim(futOI["FH_OPEN_INT"].min()*.9, futOI["FH_OPEN_INT"].max()*1.1)
    longplot = ax1.bar(x, futOI["FH_LONG"], width, color='green')
    longuplot = ax1.bar(x, futOI["FH_LONGU"], width, color='cyan',alpha = 0.5)
    shortplot = ax1.bar(x, futOI["FH_SHORT"], width, color='red')
    longuplot = ax1.bar(x, futOI["FH_SHORTC"], width, color='orange',alpha = 0.6)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Open Interest')
    ax1.set_title('Fut Open Interest')
    ax1.set_xticks(x, date,rotation ='vertical')

    ax2 = ax1.twinx()
    ax2.plot(x, futOI["FH_SETTLE_PRICE"], color='black', label='price')


    fig.tight_layout()
    
    return fig

def check_Breakout(row):
    breakout = "Avoid"
    if (row['Close'] > row['SMA 100'] and row['Close'] > row['SMA 50'] and row['Close'] > row['SMA 20'] and row['Close'] > row['SMA 10'] and row['Close'] > row['SMA 5'] and row['Close'] < row['SMA 200']):
        breakout = "Potential Long"
    if (row['Close'] < row['SMA 100'] and row['Close'] < row['SMA 50'] and row['Close'] < row['SMA 20'] and row['Close'] < row['SMA 10'] and row['Close'] < row['SMA 5'] and row['Close'] > row['SMA 200']):
        breakout = "Potential Short"
    return breakout

# Set Page Ttitle
st.set_page_config(page_title = "Options Dashboard", page_icon= ":bar_chart:", layout="wide")

#st.dataframe(nseFNOSecurities_df)

# Set Sidebar details

st.sidebar.title("FNO Anlaysis")

AnalysisType = st.sidebar.selectbox("Choose Analysis Type", ('Overview','Intra OI Trigger','Option Chain','Open Interest'))
st.sidebar.markdown("----")

if AnalysisType == 'Option Chain':

   Instrument = st.sidebar.selectbox("Select FNO Stock", (nseFNOSecurities_df1["ExchangeCode"]))

if AnalysisType == 'Open Interest':

   Instrument = st.sidebar.selectbox("Select FNO Stock", (nseFNOSecurities_df["ExchangeCode"]))


# Set MainPage Detail 

# Main Page Landing page - if Analysis Type is Blank
# Nifty Value, BNF Value, Top OI Gainer and Losser 
if AnalysisType == 'Overview':
    url = 'https://www.topstockresearch.com/rt/IndexAnalyser/Nifty200/Technicals/SMA'
    session = requests.Session()
    r = session.get(url,headers=headers,timeout=10)

    breakoutDF = pd.read_html(r.text)[0].dropna()
    breakoutDF.drop(breakoutDF[breakoutDF['Name'].astype(str).str[0] == '('].index, inplace = True)
    breakoutDF[['Close', 'SMA 5', 'SMA 10', 'SMA 15', 'SMA 20', 'SMA 50', 'SMA 100', 'SMA 200']] = breakoutDF[['Close', 'SMA 5', 'SMA 10', 'SMA 15', 'SMA 20', 'SMA 50', 'SMA 100', 'SMA 200']].astype('double').astype('int')

    # breakoutDF.reset_index(drop=True, inplace=True)
    breakoutDF["Super_Breakout"] = breakoutDF.apply (lambda row: check_Breakout(row), axis=1)
    
    st.header ("Super Break Out stocks")
    col1,col2 = st.columns(2)
    col1.subheader ("Long Stocks")
    col1.dataframe( breakoutDF[breakoutDF["Super_Breakout"]=="Potential Long"][["Name","Close","SMA 200"]])
    col2.subheader ("Short Stocks")
    col2.dataframe( breakoutDF[breakoutDF["Super_Breakout"]=="Potential Short"][["Name","Close","SMA 200"]] )
    
    
# Main page load for Intra OI Trigger View 
if AnalysisType == 'Intra OI Trigger':
    Trigger_load_state = st.text("Loading Data...Wait...")
    
    ceTrigger, peTrigger = getTriggerData()
    ceTrigger = ceTrigger.astype({'LTP': 'int', 'CE_STRIKE': 'int'})
    peTrigger = peTrigger.astype({'LTP': 'int', 'PE_STRIKE': 'int'})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header ("CE Triggers")   
        st.dataframe(ceTrigger.style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'}).format(precision=1),600,600)
    with col2:
        st.header ("PE Triggers")
        st.dataframe(peTrigger.style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'}).format(precision=1),600,600)
    Trigger_load_state.text("Data Loaded")

# Main Page Load for Future OI Dashboard

# Main Page Load for Options Dashboard for an instrument 
if AnalysisType == 'Option Chain':
    
    original_title = '<p style="font-family:Calibri; color:Black; text-align:center; font-size: 40px;">Option Data for - {}</p>'.format(Instrument)
    
    option_chain_data = option_chain(Instrument)
    option_chain = option_chain_data[["CE_OICHG", "CE_OI", "STRIKE", "PE_OI", "PE_OICHG"]].sort_values(by=['STRIKE'],ascending = False)
    #st.title("Option Data for - {}".format(Instrument
    st.markdown(original_title,unsafe_allow_html=True)
    st.subheader("Current Market Price - {}".format(option_chain_data["LTP"].max()))
    chain_fig = get_option_plot(option_chain_data)
    
    col1, col2 = st.columns([2,1])       
    col1.subheader ("Option Chain Chart")
    col1.pyplot(chain_fig)
    col2.subheader ("Option Chain")
    col2.dataframe(option_chain)

if AnalysisType == 'Open Interest':
    
    original_title = '<p style="font-family:Calibri; color:Black; text-align:center; font-size: 40px;">Option Interest for - {}</p>'.format(Instrument)
    open_interest_data = open_interest(Instrument,30)
    st.markdown(original_title,unsafe_allow_html=True)
    
    OI_fig = get_OI_plot(open_interest_data)
    
    col1, col2 = st.columns([4,1])       
    col1.subheader ("Option Chain Chart")
    col1.pyplot(OI_fig)
    