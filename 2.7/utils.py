import tensorflow.compat.v1 as tf

########## You have to replace the line above by those two if you have tensorflow > 1.X
# tf.compat.v1.disable_eager_execution()
# tf.disable_v2_behavior()
##########
import numpy as np
import librosa
import sys
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')


############################################################################# SIGNAL PROCESSING PART ####################################################################

def cmvn_slide(feat,winlen=300,cmvn=False): #feat : (length, dim) 2d matrix
# function for Cepstral Mean Variance Normalization

    maxlen = np.shape(feat)[0]
    new_feat = np.empty_like(feat)
    cur = 1
    leftwin = 0
    rightwin = winlen/2
    
    # middle
    for cur in range(maxlen):
        cur_slide = feat[cur-leftwin:cur+rightwin,:] 
        #cur_slide = feat[cur-winlen/2:cur+winlen/2,:]
        mean =np.mean(cur_slide,axis=0)
        std = np.std(cur_slide,axis=0)
        if cmvn == 'mv':
            new_feat[cur,:] = (feat[cur,:]-mean)/std # for cmvn        
        elif cmvn =='m':
            new_feat[cur,:] = (feat[cur,:]-mean) # for cmn
        if leftwin<winlen/2:
            leftwin+=1
        elif maxlen-cur < winlen/2:
            rightwin-=1    
    return new_feat


def feat_extract(filelist,feat_type,n_fft_length=512,hop=160,vad=True,cmvn=False,exclude_short=500):
# function for feature extracting
    sys.path.insert(0, './data')

    feat = []
    utt_shape = []
    new_utt_label =[]
    for index,wavname in enumerate(filelist):
        #read audio input
        y, sr = librosa.core.load(wavname,sr=16000,mono=True,dtype='float')

        #extract feature
        if feat_type =='melspec':
            Y = librosa.feature.melspectrogram(y,sr,n_fft=n_fft_length,hop_length=hop,n_mels=40,fmin=133,fmax=6955)
        elif feat_type =='mfcc':
            Y = librosa.feature.mfcc(y,sr,n_fft=n_fft_length,hop_length=hop,n_mfcc=40,fmin=133,fmax=6955)
        elif feat_type =='spec':
            Y = np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) )
        elif feat_type =='logspec':
            Y = np.log( np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) ) )
        elif feat_type =='logmel':
            Y = np.log( librosa.feature.melspectrogram(y,sr,n_fft=n_fft_length,hop_length=hop,n_mels=40,fmin=133,fmax=6955) )

        Y = Y.transpose()
        
        
        # Simple VAD based on the energy
        if vad:
            E = librosa.feature.rmse(y, frame_length=n_fft_length,hop_length=hop,)
            threshold= np.mean(E)/2 * 1.04
            vad_segments = np.nonzero(E>threshold)
            if vad_segments[1].size!=0:
                Y = Y[vad_segments[1],:]

                
        #exclude short utterance under "exclude_short" value
        if exclude_short == 0 or (Y.shape[0] > exclude_short):
            if cmvn:
                Y = cmvn_slide(Y,300,cmvn)
            feat.append(Y)
            utt_shape.append(np.array(Y.shape))
#             new_utt_label.append(utt_label[index])
            sys.stdout.write('%s\r' % index)
            sys.stdout.flush()
            
            
        #If you uncomment this, you will only treat your FIRST file    
#         if index ==0:
#             break

        
    tffilename = feat_type+'_fft'+str(n_fft_length)+'_hop'+str(hop)
    if vad:
        tffilename += '_vad'
    if cmvn=='m':
        tffilename += '_cmn'
    elif cmvn =='mv':
        tffilename += '_cmvn'
    if exclude_short >0:
        tffilename += '_exshort'+str(exclude_short)

    return feat, new_utt_label, utt_shape, tffilename #feat : (length, dim) 2d matrix



########################################################## MACHINE LEARNING PART #########################################################################################

def initiate_machine_learning_architecture(is_training=False, is_batchnorm=False,input_dim=40):
    import e2e_model as nn_model_foreval
    
    softmax_num = 5
    x = tf.placeholder(tf.float32, [None,None,40])
    y = tf.placeholder(tf.int32, [None])
    s = tf.placeholder(tf.int32, [None,2])

    emnet_validation = nn_model_foreval.nn(x,y,y,s,softmax_num, is_training, input_dim, is_batchnorm)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer()

    ### Loading neural network 
    saver.restore(sess,'./data/pretrained_model/model1284000.ckpt-1284000')
    return emnet_validation, x, y, s

def run_on_dataset(filename,feat_type, show=False):
    import time
    # Feature extraction configuration
    #FEAT_TYPE = 'logmel'
    N_FFT = 400
    HOP = 160
    VAD = True
    CMVN = 'mv'
    EXCLUDE_SHORT=0

    # extracting mfcc for input wavfile
    #FILENAME = ['data/EGY000001.wav','data/GLF000001.wav','data/LAV000001.wav']
    start_time = time.time()

    feat, utt_label, utt_shape, tffilename = feat_extract(filename,feat_type,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT)

    elapsed_time = time.time() - start_time
    print(format(elapsed_time) + ' seconds')


    start_time = time.time()

    emnet_validation,x,y,s=initiate_machine_learning_architecture()

    results=[]
    for i,f in enumerate(feat):
        likelihood= emnet_validation.o1.eval({x:[feat[i]], s:[utt_shape[i]]})
        results.append(np.argmax(likelihood))


    elapsed_time = time.time() - start_time
    print(format(elapsed_time) + ' seconds')

    # Print out the results in barplot

    # for i,f in enumerate(FILENAME):
    #     print 'The Input wav file '+ f +' is'
    #     print languages.keys()[languages.values().index(results[i])]
        


    ######### If you want to have a nice plot of the first result which will give you the likelihood of the file belonging to each language.
    if show == True:
        import matplotlib.pyplot as plt; plt.rcdefaults()
        import matplotlib.pyplot as plt
        #matplotlib.use('Agg')
        #%matplotlib inline

        x = np.arange(5)
        dialects = ['Egyptian','Gulf','Levantine','MSA','North Africa']
        plt.bar(x,likelihood[0],align='center')
        plt.xticks(x,dialects)
        plt.ylabel('Likelihood')

        plt.title('Dialect identification offline test result')
        plt.show()
    return results