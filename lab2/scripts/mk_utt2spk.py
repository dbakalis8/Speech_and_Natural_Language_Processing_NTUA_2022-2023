folders = ['train', 'dev', 'test']

for folder in folders:
    with open('data/' + folder + '/uttids', 'r') as file:
        uttids = file.readlines()

    with open('data/' + folder + '/utt2spk', 'w') as file:
        for uttid in uttids:
            file.write(uttid[:-1] + ' ' + uttid[:2] +'\n')
