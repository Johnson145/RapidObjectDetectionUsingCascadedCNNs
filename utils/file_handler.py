import os
import platform
import socket
import subprocess
from random import choice
from typing import List
from urllib.request import Request, urlopen

#######################################################################
# allow downloading stuff with a random user agent
# see http://wolfprojects.altervista.org/articles/change-urllib-user-agent/
user_agents = [
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246',
    'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1'
]

# set default time out
timeout_seconds = 5
socket.setdefaulttimeout(timeout_seconds)


def read_txt_lines(file_path: str) -> List[str]:
    """Read the txt file at the given path and return its lines in a list."""
    with open(file_path) as f:
        lines = f.readlines()
    # remove \n symbols at the end of each line
    lines = [x.strip() for x in lines]
    return lines


def _open_url(url: str):
    req = Request(url, headers={'User-Agent': choice(user_agents)})
    # note, we are using a really large timeout of large seconds, because the ImageNet API is really slow.
    result = urlopen(req, timeout=3000)
    return result


def read_url(url: str):
    """Anonymously read the file at the given url."""
    result = _open_url(url).read()
    return result


def read_txt_url_to_str(url: str) -> str:
    """Read the entire text file at the given url into a single string."""
    result_str = read_url(url).decode('utf-8')
    return result_str


def read_txt_url_lines_to_list(url: str) -> str:
    """Read the text file at the given url into a list containing all lines as separate elements."""
    result_str = read_txt_url_to_str(url)
    lines = result_str.split("\n")
    lines = [x.strip() for x in lines]
    return lines


def open_file(path):
    """Open a given file or folder in the explorer/nautilus/..
    See https://stackoverflow.com/a/16204023/1665966
    """
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
