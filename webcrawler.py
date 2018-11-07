# -*- coding: utf-8 -*-
"""
a change had been made to test git command
"""

"""
web_crawler_stack.py
    - The web crawler with a stack.
@author: Dongwook Shin and Yash Kanoria, 2014/08/14
@author: Kriste Krstovski, 2018/08/20 (edits to reflect Python 3 version)
This web crawler walks through the URLs in the source website and keep
crawling until there is no URL to be scanned. It stops crawling if a preset
maximum number of urls have been visited.

"""
from bs4 import BeautifulSoup
import re, urllib.parse, urllib.request

url="http://www.baidu.com/" # initial URL
maxNumUrl=50 # maximum number of URLs to visit 
keywords=['finance', 'engineering', 'business', 'research']

urls = {url:1} #urls to crawl
seen = {url:1} #urls seen so far
opened = []


print("Starting with url="+str(url))
while len(urls) > 0 and len(opened) < maxNumUrl:
    # DEQUEUE A URL FROM urls AND TRY TO OPEN AND READ IT
    try:
        curr_url=max(urls.items(), key=lambda x: x[1])[0]
        urls.pop(curr_url)
        #print("num. of URLs in stack: %d " % len(urls))
        webpage=urllib.request.urlopen(curr_url)
        opened.append(curr_url)

    except Exception as ex: #if urlopen() fails
        print(ex)
        continue    #skip code below

    # IF URL OPENS, CHECK WHICH URLS THE PAGE CONTAINS
    # ADD THE URLS FOUND TO THE QUEUE url AND seen
    soup = BeautifulSoup(webpage, "html5lib")  #creates object soup
    htmltext=soup.get_text()
    score=sum([(htmltext.count(i)) for i in keywords])
    seen[curr_url]=score
    # Put child URLs into the stack
    if (score>0):
        for tag in soup.find_all('a', href = True): #find tags with links
            childUrl = tag['href']          #extract just the link
            original_childurl = childUrl
            childUrl = urllib.parse.urljoin(url, childUrl)
            #print("url=" + url)
            #print("original_childurl=" + original_childurl)
            #print("childurl=" + childUrl)
            #print("url in childUrl=" + str(url in childUrl))
            #print("childUrl not in seen=" + str(childUrl not in seen))
            if url in childUrl and childUrl not in seen:
                #print("***urls.append and seen.append***")
                urls[childUrl]=score
                seen[childUrl]=0
            else:
                print("######")


print("num. of URLs seen = %d, and scanned = %d" % (len(seen), len(opened)))

print("List of seen URLs(top 25):")
d=sorted(seen.items(),key=lambda k:k[1],reverse=True)[:25]
for seen_url in d:
    print(seen_url)