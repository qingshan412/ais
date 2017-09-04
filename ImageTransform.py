import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict
    
def load_databatch(data_folder, idx, img_size=32, train='train'):

    if train != 'train':
        data_file = os.path.join(data_folder, train + str(img_size), 'val_data')
    else:
        data_file = os.path.join(data_folder, train + str(img_size),'train_data_batch_')
    print(data_file)
    #

    d = unpickle(data_file + str(idx))
    x = d['data']
    #y = d['labels']
    #mean_image = d['mean']

    x = x/np.float32(256)#255
    #mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    #y = [i-1 for i in y]
    data_size = x.shape[0]

    #x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    #X_train = x[0:data_size, :, :, :]
    #Y_train = y[0:data_size]
    #X_train_flip = X_train[:, :, :, ::-1]
    #Y_train_flip = Y_train
    #X_train = np.concatenate((X_train, X_train_flip), axis=0)
    #Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    Px = np.sum(x,axis=1)
    line1 = Px.reshape((data_size,-1))
    line2 = line1 + np.random.random(line1.shape)/256.0 #add noise

    return line2#dict(
        #X_train=lasagne.utils.floatX(X_train),
        #Y_train=Y_train.astype('int32'),
        #mean=mean_image)

PicPath = '../DataImageNet/Image32'
### train32
#AllPx = load_databatch(data_folder = PicPath, idx = 1)
#for i in xrange(9):
#    print(i)
#    tmpPx = load_databatch(data_folder = PicPath, idx = i+2)
#    AllPx = np.vstack((AllPx, tmpPx))
#
#np.save('AllPx_train.npy', AllPx)

### valid32
AllPx = load_databatch(data_folder = PicPath, idx = 0, train = 'valid')
np.save(os.path.join(PicPath, 'AllPx_valid.npy'), AllPx)