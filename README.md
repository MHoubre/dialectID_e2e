#Acknowledgments
This project is a fork from the projet DialectID_e2e by Suwon Shon and Ahmed Ali and James Glass. Their algorithm is explained in the article "Convolutional Neural Network and Language Embeddings for End-to-End Dialect Recognition".

*Repository link:https://github.com/swshon/dialectID_e2e*

@inproceedings{Shon2018,
  author={Suwon Shon and Ahmed Ali and James Glass},
  title={Convolutional Neural Network and Language Embeddings for End-to-End Dialect Recognition	},
  year=2018,
  booktitle={Proc. Odyssey 2018 The Speaker and Language Recognition Workshop},
  pages={98--104},
  doi={10.21437/Odyssey.2018-14},
  url={http://dx.doi.org/10.21437/Odyssey.2018-14}
}

You will find the original README from the base project in the 2.7 directory.


#Goal of this project

I found this algorithm very interesting but because of its implementation in Python 2, I wanted to update it in Python 3 and using Tensorflow 2.
But first I also wanted to use it as a real tool and evaluate it on my datasets. Or the run.sh only runs to augment datasets and train different models (which is already a great thing). So I added a few scripts which are basically a reorganisation of the offline_test jupyter notebook and making it usable on an entire dataset rather than on a single file.

#How to use it
For now, to use the solution, you will just have to put all your wav files in a "data" folder in the 2.7 folder.
Then you run main.py
If you want a report because you tested it for expertise, your labels have to be present in the title of the file (for example "__EGY1.wav__") and the function __get_references__ will fill the reference list for the __sklearn.metrics.classification_report__.
