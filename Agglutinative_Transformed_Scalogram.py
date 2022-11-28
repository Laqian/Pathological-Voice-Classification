import os

voice = []


def read_voice_data(path):
    voice_data = []
    file = open(path)
    line = file.readline()
    while line:
        voice_data.append(float(line))
        line = file.readline()
    return voice_data


if __name__ == '__main__':
    data_path = '../Data/VOICED'
    files = os.listdir(data_path)
    for name in files:
        if name[:5] == "voice" and name[-3:] == "txt":
            if name[8] != "-":
                # print(name[:-4])
                # print(read_voice_data(data_path + '/' + name))
                voice.append(read_voice_data(data_path + '/' + name))
