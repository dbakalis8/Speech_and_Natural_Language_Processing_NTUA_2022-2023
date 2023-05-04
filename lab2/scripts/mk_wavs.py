folders = ['train', 'dev', 'test']

for folder in folders:
    with open('data/' + folder + '/uttids', 'r') as file:
        uttids = file.readlines()

    with open('data/' + folder + '/wav.scp', 'w') as file:
        for uttid in uttids:
            file.write(uttid[:-1] + ' ' + './wav/' + uttid[:-1] +'.wav\n')
