import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/Bilal/Desktop/New folder/SMS SPAM/spam.csv", encoding='ISO-8859-1', usecols=[0, 1])
data.head()
data.shape
data.describe
data.v1.value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='v1',data=data,palette='Blues')
plt.title('Distribution Of Target',fontsize=20)
plt.xlabel('Target',fontsize=16)
plt.ylabel('Count',fontsize=16)
plt.grid(True)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
Target=data.v1
Target
Target.replace('spam',1,inplace=True)
Target.replace('ham',0,inplace=True)
Target
#Transform Output
Target.replace('spam',1,inplace=True)
Target.replace('ham',0,inplace=True)
Target
//pip install wordcloud
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
plt.figure(figsize=(30,20))
plt.imshow(WordCloud(background_color = 'white').generate(" ".join(data.v2)))
plt.axis("off")
plt.title("WordCloud For Text Before StopWords",fontsize=20)
plt.show()
Text=data.v2
Text
#Split Data
X_train,X_test,y_train,y_test=train_test_split(Text,Target,test_size=.2,shuffle=True,random_state=44)
print('X_train Shape :',X_train.shape)
print('X_test Shape :',X_test.shape)
print('y_train Shape :',y_train.shape)
print('y_test Shape :',y_test.shape)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#Model
model=Pipeline([
    ('tfid',TfidfVectorizer()),
    ('model',MultinomialNB(alpha=.1))
    ])
model.fit(X_train,y_train)
print('Pipeline Model Train Score is : ' , model.score(X_train, y_train))
print('Pipeline Model Test Score is : ' , model.score(X_test, y_test))
y_pred=model.predict(X_test)
df=pd.DataFrame()
df['Actual'],df['Predicted']=y_test,y_pred
df
CM = confusion_matrix(y_test, y_pred)
sns.heatmap(CM,annot=True,fmt='g',center = True,cmap='Blues')
plt.show()
CM
ClassificationReport = classification_report(y_test,y_pred)
print('Classification Report is : ', ClassificationReport )
!pip install -U gensim
import gensim.downloader as api
wiki_embeddings = api.load('glove-wiki-gigaword-100')
from gensim.models import Word2Vec
wiki_embeddings['king']
wiki_embeddings.most_similar('king')
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 100)
messages = pd.read_csv("C:/Users/Bilal/Desktop/New folder/SMS SPAM/spam.csv", encoding='latin-1')
messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages.columns = ["label", "text"]
messages.head()
messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
messages.head()
X_train, X_test, y_train, y_test = train_test_split(messages['text_clean'], messages['label'],test_size = 0.2)
#Train the word2vec model
w2v_model = gensim.models.Word2Vec(X_train,
                            vector_size=100,
                            window=5,
                            min_count=2)
w2v_model.wv['king']
w2v_model.wv.most_similar('king')
for i, v in enumerate(w2v_vect):
    print(len(X_test.iloc[i]), len(v))

# CodSoft
