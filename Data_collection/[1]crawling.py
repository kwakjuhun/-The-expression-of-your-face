from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

import json, os, argparse, sys, requests
import urllib, urllib3, datetime, time
from urllib3.exceptions import InsecureRequestWarning

urllib3.disable_warnings(InsecureRequestWarning)

searchword = 'str'
chromedriver = 'chromedriver_path'
dirs = 'dirs_path' 

searchurl = 'https://www.google.com/search?q=' +searchword+ '&source=lnms&tbm=isch'
maxcount = 1000


if not os.path.exists(dirs):
    os.mkdir(dirs)

def download_google_staticimages():
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')

    try:
        browser = webdriver.Chrome(chromedriver, options=options)
    except Exception as e:
        sys.exit()

    browser.set_window_size(1280, 1024)
    browser.get(searchurl)
    time.sleep(1)

    element = browser.find_element_by_tag_name('body')

    for i in range(50):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.find_element_by_id('smb').click()
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)
    except:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)

    time.sleep(0.5)

    browser.find_element_by_xpath('//input[@value="결과 더보기"]').click()

    for i in range(50):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.find_element_by_id('smb').click()
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)
    except:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)

    page_source = browser.page_source 

    soup = BeautifulSoup(page_source, 'lxml')
    images = soup.find_all('img')

    urls = []
    for image in images:
        try:
            url = image['data-src']
            if not url.find('https://'):
                urls.append(url)
        except:
            try:
                url = image['src']
                if not url.find('https://'):
                    urls.append(image['src'])
            except Exception as e:
                print(e)

    count = 0
    if urls:
        for url in urls:
            try:
                res = requests.get(url, verify=False, stream=True)
                rawdata = res.raw.read()
                with open(os.path.join(dirs, 'img_' + str(count) + '.jpg'), 'wb') as f:
                    f.write(rawdata)
                    count += 1
            except Exception as e:
                print(e)

    browser.close()
    return count

def main():
    count = download_google_staticimages()
    print('end')

if __name__ == '__main__':
    main()