source path.sh

feat-to-len scp:data/train/feats.scp ark,t:data/train/feats_len

head -5 data/train/feats_len

feat-to-dim ark:mfcc_train/raw_mfcc_train.1.ark - 
