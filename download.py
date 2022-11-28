import urllib
from urllib.request import urlopen


def download(url_str, save_path, filename):
    full_path = save_path + '/' + filename
    try:
        urllib.request.urlretrieve(url_str, full_path)
        print("download success")
        return full_path
    except:
        print("failed")
        return False


if __name__ == '__main__':
    path1 = '../Data/VOICED/voice'
    path2 = '../Data/VOICED/label'
    for i in range(1, 209):
        index = "{:0>3d}".format(i)
        url = 'https://physionet.org/files/voiced/1.0.0/voice'+ index +'.txt?download'
        download(url, path1, "voice"+index+'.txt')
    for i in range(1, 209):
        index = "{:0>3d}".format(i)
        url = 'https://physionet.org/files/voiced/1.0.0/voice'+ index +'-info.txt?download'
        download(url, path2, "voice"+index+'-info.txt')