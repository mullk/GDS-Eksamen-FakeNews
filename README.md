# GDS-Eksamen-FakeNews
Our exam project: 

Data_Processing can be called using:    python Data_Processing <PathToCSV> <optional row limit>

Example usage:    python Data_Processing.py Data\news_sample.csv
Example usage:    python Data_Processing.py Data\995,000_rows.csv 1000
Example usage:    python split_data.py Data/news_sample.csv

The results are documented in our jupyter notebooks:

part_1.ipynb: Includes data processing, splitting etc.
part_2.ipynb: The simple model (logistic regression) and test on the LIAR dataset. 
Del 3.ipynb:  Contains the advanced model and training of it.

The notebooks contain the outputs, and results of the advanced model can also be found in the folder "Data_Opgave_3"" in our repository as well.








# Sources


Python standard library: https://docs.python.org/3/

re regex-dokumentation: https://docs.python.org/3/library/re

Regex HOWTO: https://docs.python.org/3/howto/regex

csv-modulet: https://docs.python.org/3/library/csv

pathlib til filstier: https://docs.python.org/3/library/pathlib


pandas.read_csv: https://pandas.pydata.org/docs/reference/api/pandas.read_csv

Pandas IO guide: https://pandas.pydata.org/docs/user_guide/io

Pandas scaling to large datasets: https://pandas.pydata.org/pandas-docs/stable/user_guide/scale


NLTK hovedside: https://www.nltk.org/

NLTK tokenizers: https://www.nltk.org/api/nltk.tokenize

NLTK data installation: https://www.nltk.org/data

SnowballStemmer API: https://www.nltk.org/api/nltk.stem.SnowballStemmer

Stemming examples/how-to: https://www.nltk.org/howto/stem


Feature extraction overview: https://scikit-learn.org/stable/modules/feature_extraction

Text feature extraction section: https://scikit-learn.org/stable/modules/feature_extraction

TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer

TfidfTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer

Working with text data tutorial: https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data


Scikit-learn getting started: https://scikit-learn.org/stable/getting_started

LinearSVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC

SVM user guide: https://scikit-learn.org/stable/modules/svm

SGDClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier

SGD user guide: https://scikit-learn.org/stable/modules/sgd

Linear models overview: https://scikit-learn.org/stable/modules/linear_model


Feature selection overview: https://scikit-learn.org/stable/modules/feature_selection

chi2 feature scoring: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2

SequentialFeatureSelector: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector

TruncatedSVD: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD

Decomposition guide / truncated SVD: https://scikit-learn.org/stable/modules/decomposition


Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline

Pipeline/composite estimators guide: https://scikit-learn.org/stable/modules/compose

GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV

Hyperparameter tuning guide: https://scikit-learn.org/stable/modules/grid_search

Cross-validation guide: https://scikit-learn.org/stable/modules/cross_validation


train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split

Common pitfalls / leakage and Leakage guide: https://scikit-learn.org/stable/common_pitfalls

classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report

f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score

confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix

ConfusionMatrixDisplay: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay

Model evaluation guide: https://scikit-learn.org/stable/modules/model_evaluation


scipy.sparse.hstack: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack


Joblib hovedside: https://joblib.readthedocs.io/en/stable/

joblib.dump and joblib.load: https://joblib.readthedocs.io/en/latest/generated/joblib.dump

Joblib persistence guide: https://joblib.readthedocs.io/en/latest/persistence

Joblib parallel guide: https://joblib.readthedocs.io/en/latest/parallel


Jupyter docs: https://docs.jupyter.org/en/latest/

Jupyter Notebook interface: https://docs.jupyter.org/en/stable/install

Start/try Jupyter: https://docs.jupyter.org/en/stable/start/


Matplotlib quick start: https://matplotlib.org/stable/users/explain/quick_start

Pyplot tutorial: https://matplotlib.org/stable/tutorials/pyplot

Matplotlib tutorials: https://matplotlib.org/stable/tutorials/index


Git main docs: https://git-scm.com/

Git reference: https://git-scm.com/docs

Git tutorial: https://git-scm.com/docs/gittutorial

Pro Git book: https://git-scm.com/book/en/v2


Pytest getting started: https://docs.pytest.org/en/stable/getting-started

Pytest docs: https://docs.pytest.org/en/stable/

Pytest how-to: https://docs.pytest.org/en/stable/how-to/index


LIAR paper/datasæt: https://aclanthology.org/P17-2067/

FakeNewsCorpus GitHub: https://github.com/several27/FakeNewsCorpus

FakeNewsCorpus helper files til KU-kontekst: https://github.com/datalogisk-fagraad/FakeNewsCorpusFiles

FakeNewsNet: https://github.com/KaiDMML/FakeNewsNet

Wang & Manning, “Baselines and Bigrams”: https://aclanthology.org/P12-2018/


TfidfVectorizer → LinearSVC → classification_report → confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer

train_test_split + leakage/common pitfalls + Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split

chi2, TruncatedSVD, GridSearchCV, SGDClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2





## Articles about FakeNews Detection we have read/scribbed


https://arxiv.org/pdf/1905.04749

https://www.nature.com/articles/s41598-025-29942-y

https://arxiv.org/html/2412.14276v1

https://arxiv.org/html/2512.18533

https://etasr.com/index.php/ETASR/article/view/10678

https://scikit-learn.org/stable/modules/naive_bayes.html

https://www.kaggle.com/code/therealsampat/fake-news-detection

https://keras.io/examples/nlp/text_classification_from_scratch/

https://www.itm-conferences.org/articles/itmconf/pdf/2023/06/itmconf_icdsac2023_03005.pdf

https://www.mdpi.com/2073-431X/14/6/237

https://arxiv.org/pdf/2411.12703





## Similar projects we've looked at for inspiration

https://github.com/daniel-was-taken/Fake-News-Detection

https://hannibunny.github.io/nlpbook/06classification/FakeNewsClassification.html

https://github.com/TheSaintIndiano/Fake-News-Detection/blob/master/Fake%20News%20Detection.py

https://github.com/borankilic/fake-news-detection

https://github.com/sapnilcsecu/supervised_text_classification/blob/master/super_text_class/feature_eng/char_tf_idf.py

https://github.com/nlptown/nlp-notebooks/blob/master/Traditional%20text%20classification%20with%20Scikit-learn.ipynb

https://github.com/NikosMav/FakeNews-Classification

https://github.com/Zubair-hussain/Fake-News-Detection-using-NLP

https://github.com/raj1603chdry/Fake-News-Detection-System

https://github.com/adityaRakhecha/fake-news-detector