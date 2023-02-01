import pandas as pd
import streamlit as st
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz as graphviz
import string
from streamlit_option_menu import option_menu
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tweepy
from sklearn.utils import shuffle
import time

def main():
     # twitter API connection
     consumer_key = "gHlrtstrOp7ahRheOVRUoTZiR"
     consumer_secret = "ZKzbFIs8tzGdJ8iNeYddC5zxFJmUqtVtXI5twn02ptnaNbmfgG"
     access_token = "1578249102961348608-5B3O6jdkKMsJ5JJXHtnq3W8R4v8hPw"
     access_token_secret = "6BIoeanGw7KhHPIr9DzqiPc6HPF52AaFTi36Zj4uaT95G"
     
     auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
     auth.set_access_token( access_token , access_token_secret )
     api = tweepy.API(auth)
         
     urdu_punctuations = string.punctuation
     punctuations_list = urdu_punctuations   
     
     #empty data frame with column names
     df = pd.DataFrame(columns=["Date","User","IsVerified","Text","Likes","RT",'User_location'])
     
     #extracting tweets related to the topic
     def get_tweets(Topic,Count):    
         i=0
         for tweet in tweepy.Cursor(api.search_tweets, q=Topic,count=Count, lang='ur',exclude='retweets').items():
             # index i till count
             print(i, end='\r')
             df.loc[i,"Date"] = tweet.created_at
             df.loc[i,"User"] = tweet.user.name
             df.loc[i,"IsVerified"] = tweet.user.verified
             df.loc[i,"Text"] = tweet.text
             df.loc[i,"Likes"] = tweet.favorite_count
             df.loc[i,"RT"] = tweet.retweet_count
             df.loc[i,"User_location"] = tweet.user.location
             #for excelsheet
             #df.to_csv("TweetDataset.csv",index=False)
             #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
             i=i+1
             if i>Count:
                 break
             else:
                 pass
     #preprocessing       
  
     def cleaning_punctuations(text):
             translator = str.maketrans('', '', punctuations_list)
             return text.translate(translator)
    
     def cleaning_URLs(data):
              return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)

     def clean_number(data):
              return re.sub('[0-9]+', '', data)
      
     def clean_english_chr(data):
              return re.sub('[a-zA-Z][a-zA-Z0-9]*', '', data)
      
     def clean_special_chr(data):
              return re.sub("[$&+,:;=?@#|'<>.-^*()%!]", '', data)
     
     #reading file
     #DATASET_COLUMNS=['text','Category']
     df2 = pd.read_excel('UrduAnnotatedDataset.xlsx')
     
     shuffled_df2= shuffle(df2, random_state=1)
     #using shuffled dataframe so testing isnt dedicated to 1 topic only
     X=shuffled_df2.Text
     #Y= labels
     Y=shuffled_df2.Annotator3

     # Separating the 70% data for training data and 30% for testing data
     X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)
     #to extract features from text - both 1 ngram and 2 ngram features 
     vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
     #search ngram and word - 123 ko combine kr lain
     #best accuracy at 5000
     #fitting(header is passed to vectoriser) trainting data to vectoriser 
     vectoriser.fit(X_train)

     #transforming train and test into vectors for machine learning models
     X_train = vectoriser.transform(X_train)
     X_test  = vectoriser.transform(X_test)
     
     BNBmodel = BernoulliNB()
     #fit data(input of training data, output of training data)                
     BNBmodel.fit(X_train, y_train)
     #evaluation  - F1 score, accuracy etc
     
     SVCmodel = LinearSVC()#training on 80% of the input
     SVCmodel.fit(X_train, y_train)#fit data
     
     LRmodel = LogisticRegression()#training on 80% of the input --- default parameters removed
     LRmodel.fit(X_train, y_train)#fit data
     
     RFmodel = RandomForestClassifier()
     RFmodel.fit(X_train, y_train)
     
     @st.cache(suppress_st_warning=True)
     def get_fvalue(val):
         feature_dict = {"No":1,"Yes":2}
         for key,value in feature_dict.items():
             if val == key:
                 return value

     def get_value(val,my_dict):
         for key,value in my_dict.items():
             if val == key:
                 return value
     st.sidebar.title("Select Mode")
     app_mode = st.sidebar.radio('',['About Us','Sentiment Analysis']) #two pages
     if app_mode=='About Us':
         with st.container():
             st.subheader(":wave: Hello,")
             st.title('Real-time Extraction of Urdu tweets')  
             st.subheader("This project is designed to analyze the sentiments of tweets extracted in real-time.")
             #st.image('twitter.png', caption='Twitter', width=100, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
             st.write(
                 "The tweets are primarily categorized in Positive, Negative and Neutral sentiments with the visual representation of the results."
             )
             st.subheader("Flow of the project:")
            
             st.graphviz_chart('''
        digraph {
            Tweets_from_Twitter -> Prediction_through_machine_learning_model
            Prediction_through_machine_learning_model -> Visualization
        }
    ''') 
             st.subheader("The final year project is created by:")
             st.text("Maryam Ahmed")
             st.text("Arha Amir Azeem")
             st.text("Maheen Khalid")

     elif app_mode == 'Sentiment Analysis':
       
             st.title('Extract Urdu tweets:')   
             Topic = str()
             Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))   
             if len(Topic) > 0 :
                get_tweets(Topic , Count=500)
     
                 #st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Text)))
                st.write("Total Tweets number of Tweets: {}".format(len(df.Text)))


             with st.sidebar:
                selected = option_menu(
                     menu_title="Menu",
                     options=["Extract Tweets","Extracted Data","Accuracy","Prediction & Visualization"],
                     icons=["twitter","","","file-bar-graph"])
             if selected == "Extracted Data":
                    st.title('View Extracted Data')
                    if st.button("See the Original Extracted Data"):
                        #st.markdown(html_temp, unsafe_allow_html=True)
                        st.success("Below is the Original Extracted Data :")
                        st.write(df.head(50))
         
                    df['Text'] = df['Text'].apply(lambda x: cleaning_punctuations(x))
                    df['Text'] = df['Text'].apply(lambda x: cleaning_URLs(x))
                    df['Text'] = df['Text'].apply(lambda x: clean_number(x))
                    df['Text'] = df['Text'].apply(lambda x: clean_english_chr(x))
                    df['Text'] = df['Text'].apply(lambda x: clean_special_chr(x))
                    print(df['Text'])
          
                    if st.button("See the Cleaned Data"):
                        #st.markdown(html_temp, unsafe_allow_html=True)
                        st.success("Below is the Cleaned Data :")
                        st.write(df.head(50))  
               
             if selected == "Accuracy":
                     st.title('Accuracy of the Analysis')
                     #annotation
                     #df["Category"] = df2["Text"]
            
                     def model_Evaluate(model):
                         # Predict values for Test dataset
                         y_pred = model.predict(X_test) #inbuilt prediction
                         # Print the evaluation metrics for the dataset.
                         #st.write('Classification report: ',classification_report(y_test, y_pred))
                         st.title("Accuracy Metrics")
                         accuracy = accuracy_score(y_test, y_pred)
                         formatted_accuracy = "{:.2f}".format(accuracy)
                         st.write('Accuracy: ', formatted_accuracy)
                         f1score = f1_score(y_test, y_pred, average="macro")
                         formatted_f1score = "{:.2f}".format(f1score)
                         st.write('F1-Score: ', formatted_f1score)
                         precision = precision_score(y_test, y_pred, average="macro")
                         formatted_precision = "{:.2f}".format(precision)
                         st.write('Precision: ', formatted_precision)
                         recall = recall_score(y_test, y_pred, average="macro")
                         formatted_recall = "{:.2f}".format(recall)
                         st.write('Recall: ', formatted_recall)
                         print(classification_report(y_test, y_pred))
                         st.title("Generating A confusion matrix")
                         # Compute and plot the Confusion matrix
                         fig, ax = plt.subplots()
                         cf_matrix = confusion_matrix(y_test, y_pred) #inbuilt from sklearn
                         categories = ['Negative','Positive','Neutral']
                         group_names = ['True Neg','False Pos', 'False Neg','True Pos']
                         group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
                         labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
                         labels = np.asarray(labels).reshape(2,2)
                         sns.heatmap(cf_matrix, annot = True, cmap = 'Blues',fmt = '', xticklabels = categories, yticklabels = categories)
                         #plotting Confusion matrix
                         plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
                         plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
                         plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
                         st.pyplot(fig)
                         #we reported f1 score(it is harmonic mean of recall and precision ) because it works on imbalanced data 
     
     
                     if st.button("Generate confusion matrix for Naive Bayes"):
                         #evaluation  - F1 score, accuracy etc
                         model_Evaluate(BNBmodel)
                    
                     if st.button("Generate confusion matrix for SVM"):
                         model_Evaluate(SVCmodel)#evaluation             
                    
                     if st.button("Generate confusion matrix for Logistic Regression"):
                         model_Evaluate(LRmodel)#evaluation 
                
                     if st.button("Generate confusion matrix for Random Forest"):
                         model_Evaluate(RFmodel)#evaluation


             if selected == "Prediction & Visualization":
                     st.title('Prediction')
                     X_test3  = vectoriser.transform(df.Text)
                     y_pred3 = RFmodel.predict(X_test3)#prediction
                     df["Prediction"] = y_pred3
                     st.write(df.head(50))
                     st.title('Plots')
                     if st.button("Get Count Plot for Different Sentiments"):
                         st.success("Generating A Count Plot")
                         st.subheader(" Count Plot for Different Sentiments")
                         ax = df.groupby('Prediction').count().plot( kind='hist', title='Distribution of data',legend=False)
                         ax.set_xticks([], minor=False)
                         ax.set_xticklabels([], rotation=0)
                         fig, ax = plt.subplots()
                         text, sentiment = list(df['Text']), list(df['Prediction'])
                         #st.write(sns.countplot(df["Category"]))
                         st.write(sns.countplot(df["Prediction"]))
                         st.pyplot(fig)
            
                     if st.button("Get pie chart for Different Sentiments"):
                         st.success("Generating A Pie chart")
                         st.subheader(" Count Plot for Different Sentiments")
                         fig, ax = plt.subplots()
                         size = [len(df[df['Prediction'] == 'neg']),len(df[df['Prediction'] == 'pos']),len(df[df['Prediction'] == 'neu'])]
                         explode = (0.1, 0.0, 0.1)
                         st.write(plt.pie(size,shadow=True,explode=explode,labels=["Negative","Positive","Neutral"],autopct='%1.2f%%'))
                         st.pyplot(fig) 
                     
                     if st.button("Get Count Plot Based on Verified and unverified Users"):
                         st.success("Generating A Count Plot (Verified and unverified Users)")
                         st.subheader(" Count Plot for Different Sentiments for Verified and unverified Users")
                         fig, ax = plt.subplots()
                         st.write(sns.countplot(df["Prediction"],hue=df.IsVerified))
                         st.pyplot(fig)
               
                     if st.button("Get Count Plot based on Location"):
                         st.success("Generating A Count Plot for User Locations")
                         st.subheader(" Count Plot for Different Users tweeting based on their locations")
                         df['freq_count'] = df.groupby('User_location')['User_location'].transform('count')
                         fig, ax = plt.subplots(figsize = (25,15))
                         st.write(sns.countplot(df["freq_count"],data=df['User_location']))
                         st.pyplot(fig)

if __name__ == '__main__':
    main()
     
