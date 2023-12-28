#!/usr/bin/env python
# coding: utf-8

# In[260]:


import pandas as pd
from selenium import webdriver

PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)

urls  = [
    'https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/',
    'https://insights.blackcoffer.com/what-if-the-creation-is-taking-over-the-creator/',
    'https://insights.blackcoffer.com/what-jobs-will-robots-take-from-humans-in-the-future/',
    'https://insights.blackcoffer.com/will-machine-replace-the-human-in-the-future-of-work/',
    'https://insights.blackcoffer.com/will-ai-replace-us-or-work-with-us/',
    'https://insights.blackcoffer.com/man-and-machines-together-machines-are-more-diligent-than-humans-blackcoffe/',
    'https://insights.blackcoffer.com/in-future-or-in-upcoming-years-humans-and-machines-are-going-to-work-together-in-every-field-of-work/',
    'https://insights.blackcoffer.com/how-neural-networks-can-be-applied-in-various-areas-in-the-future/',
    'https://insights.blackcoffer.com/how-machine-learning-will-affect-your-business/',
    'https://insights.blackcoffer.com/deep-learning-impact-on-areas-of-e-learning/',
    'https://insights.blackcoffer.com/how-to-protect-future-data-and-its-privacy-blackcoffer/',
    'https://insights.blackcoffer.com/how-machines-ai-automations-and-robo-human-are-effective-in-finance-and-banking/',
    'https://insights.blackcoffer.com/ai-human-robotics-machine-future-planet-blackcoffer-thinking-jobs-workplace/',
    'https://insights.blackcoffer.com/how-ai-will-change-the-world-blackcoffer/',
    'https://insights.blackcoffer.com/future-of-work-how-ai-has-entered-the-workplace/',
    'https://insights.blackcoffer.com/ai-tool-alexa-google-assistant-finance-banking-tool-future/',
    'https://insights.blackcoffer.com/ai-healthcare-revolution-ml-technology-algorithm-google-analytics-industrialrevolution/',
    'https://insights.blackcoffer.com/all-you-need-to-know-about-online-marketing/',
    'https://insights.blackcoffer.com/evolution-of-advertising-industry/',
    'https://insights.blackcoffer.com/how-data-analytics-can-help-your-business-respond-to-the-impact-of-covid-19/',
    'https://insights.blackcoffer.com/covid-19-environmental-impact-for-the-future/',
    'https://insights.blackcoffer.com/environmental-impact-of-the-covid-19-pandemic-lesson-for-the-future/',
    'https://insights.blackcoffer.com/how-data-analytics-and-ai-are-used-to-halt-the-covid-19-pandemic/',
    'https://insights.blackcoffer.com/difference-between-artificial-intelligence-machine-learning-statistics-and-data-mining/',
    'https://insights.blackcoffer.com/how-python-became-the-first-choice-for-data-science/',
    'https://insights.blackcoffer.com/how-google-fit-measure-heart-and-respiratory-rates-using-a-phone/',
    'https://insights.blackcoffer.com/what-is-the-future-of-mobile-apps/',
    'https://insights.blackcoffer.com/impact-of-ai-in-health-and-medicine/',
    'https://insights.blackcoffer.com/telemedicine-what-patients-like-and-dislike-about-it/',
    "https://insights.blackcoffer.com/how-we-forecast-future-technologies/",
    "https://insights.blackcoffer.com/can-robots-tackle-late-life-loneliness/",
    "https://insights.blackcoffer.com/embedding-care-robots-into-society-socio-technical-considerations/",
    "https://insights.blackcoffer.com/management-challenges-for-future-digitalization-of-healthcare-services/",
    "https://insights.blackcoffer.com/are-we-any-closer-to-preventing-a-nuclear-holocaust/",
    "https://insights.blackcoffer.com/will-technology-eliminate-the-need-for-animal-testing-in-drug-development/",
    "https://insights.blackcoffer.com/will-we-ever-understand-the-nature-of-consciousness/",
    "https://insights.blackcoffer.com/will-we-ever-colonize-outer-space/",
    "https://insights.blackcoffer.com/what-is-the-chance-homo-sapiens-will-survive-for-the-next-500-years/",
    "https://insights.blackcoffer.com/why-does-your-business-need-a-chatbot/",
    "https://insights.blackcoffer.com/how-you-lead-a-project-or-a-team-without-any-technical-expertise/",
    "https://insights.blackcoffer.com/can-you-be-great-leader-without-technical-expertise/",
    "https://insights.blackcoffer.com/how-does-artificial-intelligence-affect-the-environment/",
    "https://insights.blackcoffer.com/how-to-overcome-your-fear-of-making-mistakes-2/",
    "https://insights.blackcoffer.com/is-perfection-the-greatest-enemy-of-productivity/",
    "https://insights.blackcoffer.com/global-financial-crisis-2008-causes-effects-and-its-solution/",
    "https://insights.blackcoffer.com/gender-diversity-and-equality-in-the-tech-industry/",
    "https://insights.blackcoffer.com/how-to-overcome-your-fear-of-making-mistakes/",
    "https://insights.blackcoffer.com/how-small-business-can-survive-the-coronavirus-crisis/",
    "https://insights.blackcoffer.com/impacts-of-covid-19-on-vegetable-vendors-and-food-stalls/",
    "https://insights.blackcoffer.com/impacts-of-covid-19-on-vegetable-vendors/",
    "https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-tourism-aviation-industries/",
    "https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-sports-events-around-the-world/",
    "https://insights.blackcoffer.com/changing-landscape-and-emerging-trends-in-the-indian-it-ites-industry/",
    "https://insights.blackcoffer.com/online-gaming-adolescent-online-gaming-effects-demotivated-depression-musculoskeletal-and-psychosomatic-symptoms/",
    "https://insights.blackcoffer.com/human-rights-outlook/",
    "https://insights.blackcoffer.com/how-voice-search-makes-your-business-a-successful-business/",
    "https://insights.blackcoffer.com/how-the-covid-19-crisis-is-redefining-jobs-and-services/",
    "https://insights.blackcoffer.com/how-to-increase-social-media-engagement-for-marketers/",
  "https://insights.blackcoffer.com/impacts-of-covid-19-on-streets-sides-food-stalls/",
  "https://insights.blackcoffer.com/coronavirus-impact-on-energy-markets-2/",
  "https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry-5/",
  "https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis-4/",
  "https://insights.blackcoffer.com/estimating-the-impact-of-covid-19-on-the-world-of-work-2/",
  "https://insights.blackcoffer.com/estimating-the-impact-of-covid-19-on-the-world-of-work-3/",
  "https://insights.blackcoffer.com/travel-and-tourism-outlook/",
  "https://insights.blackcoffer.com/gaming-disorder-and-effects-of-gaming-on-health/",
  "https://insights.blackcoffer.com/what-is-the-repercussion-of-the-environment-due-to-the-covid-19-pandemic-situation/",
  "https://insights.blackcoffer.com/what-is-the-repercussion-of-the-environment-due-to-the-covid-19-pandemic-situation-2/",
  "https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-office-space-and-co-working-industries/",
  "https://insights.blackcoffer.com/contribution-of-handicrafts-visual-arts-literature-in-the-indian-economy/",
  "https://insights.blackcoffer.com/how-covid-19-is-impacting-payment-preferences/",
  "https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work-2/",
  "https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis/",
  "https://insights.blackcoffer.com/covid-19-how-have-countries-been-responding/",
  "https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry-2/",
  "https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work-3/",
  "https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry-3/",
  "https://insights.blackcoffer.com/estimating-the-impact-of-covid-19-on-the-world-of-work/",
  "https://insights.blackcoffer.com/covid-19-how-have-countries-been-responding-2/",
  "https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work-4/",
  "https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis-2/",
  "https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis-3/",
  "https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry-4/",
  "https://insights.blackcoffer.com/why-scams-like-nirav-modi-happen-with-indian-banks/",
  "https://insights.blackcoffer.com/impact-of-covid-19-on-the-global-economy/",
  "https://insights.blackcoffer.com/impact-of-covid-19coronavirus-on-the-indian-economy-2/",
  "https://insights.blackcoffer.com/impact-of-covid-19-on-the-global-economy-2/",
  "https://insights.blackcoffer.com/impact-of-covid-19-coronavirus-on-the-indian-economy-3/",
  "https://insights.blackcoffer.com/should-celebrities-be-allowed-to-join-politics/",
  "https://insights.blackcoffer.com/how-prepared-is-india-to-tackle-a-possible-covid-19-outbreak/",
  "https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work/",
  "https://insights.blackcoffer.com/controversy-as-a-marketing-strategy/",
  "https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry/",
  "https://insights.blackcoffer.com/coronavirus-impact-on-energy-markets/",
  "https://insights.blackcoffer.com/what-are-the-key-policies-that-will-mitigate-the-impacts-of-covid-19-on-the-world-of-work/",
  "https://insights.blackcoffer.com/marketing-drives-results-with-a-focus-on-problems/",
  "https://insights.blackcoffer.com/continued-demand-for-sustainability/",
  "https://insights.blackcoffer.com/coronavirus-disease-covid-19-effect-the-impact-and-role-of-mass-media-during-the-pandemic/",
  "https://insights.blackcoffer.com/should-people-wear-fabric-gloves-seeking-evidence-regarding-the-differential-transfer-of-covid-19-or-coronaviruses-generally-between-surfaces/",
  "https://insights.blackcoffer.com/why-is-there-a-severe-immunological-and-inflammatory-explosion-in-those-affected-by-sarms-covid-19/",
  "https://insights.blackcoffer.com/what-do-you-think-is-the-lesson-or-lessons-to-be-learned-with-covid-19/",
  "https://insights.blackcoffer.com/coronavirus-the-unexpected-challenge-for-the-european-union/",
  "https://insights.blackcoffer.com/industrial-revolution-4-0-pros-and-cons/",
  "https://insights.blackcoffer.com/impact-of-covid-19-coronavirus-on-the-indian-economy/",
  "https://insights.blackcoffer.com/impact-of-covid-19-coronavirus-on-the-indian-economy-2/",
  "https://insights.blackcoffer.com/impact-of-covid-19coronavirus-on-the-indian-economy/",
  "https://insights.blackcoffer.com/impact-of-covid-19-coronavirus-on-the-global-economy/",
  "https://insights.blackcoffer.com/ensuring-growth-through-insurance-technology/",
  "https://insights.blackcoffer.com/blockchain-in-fintech/",
  "https://insights.blackcoffer.com/blockchain-for-payments/",
  "https://insights.blackcoffer.com/the-future-of-investing/",
  "https://insights.blackcoffer.com/big-data-analytics-in-healthcare/",
  "https://insights.blackcoffer.com/business-analytics-in-the-healthcare-industry/",
  "https://insights.blackcoffer.com/challenges-and-opportunities-of-big-data-in-healthcare/"
]




data = []  # Initialize an empty list to store the scraped data

# Loop through the URLs and scrape data
for url in urls:
    # Navigate to the webpage
    driver.get(url)
    
    # Scrape the title
    title = driver.title
    
    # Scrape the text from 'p' elements
    paragraphs = driver.find_elements_by_tag_name('p')
    text = [p.text for p in paragraphs]
    
    # Append the title and text as a tuple to the data list
    data.append((title, text))

# Close the browser
driver.quit()

# Convert the data list into a DataFrame
df = pd.DataFrame(data, columns=['Title', 'Text'])

# Print the DataFrame
print(df)
df


# In[261]:


# Concatenate all columns into a single colum
# Drop the original columns if needed
df['full']= df['Title']+ ' ' + df['Text'].str.join(' ')
df1=df.drop(['Title','Text'],axis=1)
df1


# In[262]:


import nltk
nltk.download('stopwords')


# In[263]:


from nltk.tokenize import word_tokenize
nltk.download('punkt')
# Open the file and read its contents
with open("C:/Users/jambh/Desktop/projects/selenium project/stopwords.txt", 'r') as file:
    text = file.read()
# tokenize words of stopwords given in gdrive
stopwords = word_tokenize(text)

# Function to remove stopwords from a given text
def remove_stopwords(text):
    # Tokenize the text into words
    words = word_tokenize(text)



    # Filter out the stopwords from the words
    filtered_words = [word for word in words if word not in stopwords]

    return filtered_words

# Apply the remove_stopwords function to the 'Text' column of the dataframe
df1['clean_words'] = df1['full'].apply(remove_stopwords)


def no_f_clean(x):
    
    a= ' '.join(x)
    return len(a)

# no of clean words
df1['no of clean words'] = df1['clean_words'].apply(no_f_clean)
df1


# In[264]:


def cleaned(x):
    for i in x:
        a= ' '.join(x)
    return a
        

df1['cleaned words'] = df1['clean_words'].apply(cleaned)
df1


# In[265]:



# Open the file and read its contents
with open("C:/Users/jambh/Desktop/projects/selenium project/positive.txt", 'r') as file:
    text = file.read()
# tokenize words of stopwords given in gdrive
positive = word_tokenize(text)

# Open the file and read its contents
with open("C:/Users/jambh/Desktop/projects/selenium project/negative.txt", 'r') as file:
    text = file.read()
# tokenize words of stopwords given in gdrive
negative = word_tokenize(text)


# Function to calculate positive score from a given text

def positive_score(words):
    positivescore=0
    for i in words:
        if i in positive:
            positivescore+=1
    return positivescore        


df1['POSITIVE SCORE'] = df1['clean_words'].apply(positive_score)

# Function to calculate negative score from a given text

def negative_score(words):
    negativescore=0
    for i in words:
        if i in negative:
            negativescore -=1
    return negativescore*(-1)        


df1['NEGATIVE SCORE'] = df1['clean_words'].apply(negative_score)
df1['POLARITY SCORE']=(df1['POSITIVE SCORE'] - df1['NEGATIVE SCORE'])/ (df1['POSITIVE SCORE'] + df1['NEGATIVE SCORE'] + 0.000001)
df1['SUBJECTIVITY SCORE']=(df1['POSITIVE SCORE'] + df1['NEGATIVE SCORE'])/ (df1['no of clean words'] + 0.000001)

df1=df1.drop(['no of clean words'], axis=1)
df1


# In[266]:


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Custom transformer to remove stopwords and punctuation from text columns in a DataFrame
class StopwordPunctuationRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.stop_words = set(stopwords.words('english'))
        self.punctuations = set(string.punctuation)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].apply(self._remove_stopwords_punctuation)
        return X
    
    def _remove_stopwords_punctuation(self, text):
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if (token.lower() not in self.stop_words) and (token not in self.punctuations)]
        return ' '.join(filtered_tokens)

# Example DataFrame

# Define the columns to apply stopwords and punctuation removal
text_columns = ['cleaned words']

# Create the pipeline
pipeline = Pipeline([
    ('stopword_punctuation_remover', StopwordPunctuationRemover(columns=text_columns))
])

# Apply the pipeline to the DataFrame
df_transformed = pipeline.transform(df1)

# Print the transformed DataFrame
df_transformed


# In[267]:


words=[]
for i in df_transformed['cleaned words']:
    words.append(len(i))

word_count=pd.DataFrame(words,columns=['WORD COUNT'])
word_count


# In[268]:


df = pd.concat([word_count, df_transformed],axis=1)

df["Character_Count_Sum"] = df["full"].apply(lambda x: sum(len(word) for word in x.split()))
df["Average Word Length"] =  df['Character_Count_Sum']/df['WORD COUNT'] 
df


# In[269]:


pip install pyphen


# In[270]:


import pandas as pd
import pyphen

# Create a Pyphen object for syllable hyphenation
dic = pyphen.Pyphen(lang='en')


# Function to count the number of syllables in a sentence
def count_syllables(sentence):
    words = sentence.split()
    syllable_count = 0
    for word in words:
        hyphenated_word = dic.inserted(word)
        syllables = hyphenated_word.split('-')
        if hyphenated_word.endswith(('es', 'ed')):
            syllable_count += len(syllables) - 1
        else:
            syllable_count += len(syllables)
    return syllable_count

# Add a column with the count of syllables for each sentence
df["Syllable_Count"] = df["full"].apply(count_syllables)

# Print the updated DataFrame
df


# In[271]:


import pandas as pd
import pyphen

# Create a Pyphen object for syllable hyphenation
dic = pyphen.Pyphen(lang='en')


# Function to count the number of syllables in a word
def count_syllables(word):
    hyphenated_word = dic.inserted(word)
    syllables = hyphenated_word.split('-')
    if hyphenated_word.endswith(('es', 'ed')):
        return len(syllables) - 1
    else:
        return len(syllables)

# Function to count the number of complex words in a sentence
def count_complex_words(sentence):
    words = sentence.split()
    complex_word_count = 0
    for word in words:
        syllable_count = count_syllables(word)
        if syllable_count > 1:
            complex_word_count += 1
    return complex_word_count

# Add a column with the count of complex words for each sentence
df["Complex_Word_Count"] = df["cleaned words"].apply(count_complex_words)

#df = df.drop('full', axis=1)
df


# In[286]:


import re

# Function to count the number of sentences in a text
def count_sentences(text):
    sentences = re.split(r"\.\s|\.\n", text)
    return len(sentences)

# Add a column with the count of sentences for each row
df["Number_of_Sentences"] = df["full"].apply(count_sentences)

import pandas as pd
import re


# Function to count the number of personal pronouns in a text (excluding "US")
def count_personal_pronouns(text):
    pronouns = re.findall(r"\b(he|him|his|she|her|hers|they|them|their|theirs)\b", text, flags=re.IGNORECASE)
    pronouns = [pronoun for pronoun in pronouns if pronoun != "US"]
    return len(pronouns)

# Add a column with the count of personal pronouns (excluding "US") for each row
df["PERSONAL PRONOUNS"] = df["full"].apply(count_personal_pronouns)
#df1 = df1.drop('full','cleaned words','clean_words', axis=1)
# Print the updated DataFrame
#df2 = pd.concat([df1, df],axis=1)

df2=df.drop(['clean_words','cleaned words','Character_Count_Sum'],axis=1)
df2


# In[287]:


def total_words(x):
    return len(x)

df2['TOTAL NO OF WORDS']=df['full'].apply(total_words)
df2


# In[288]:


# Average Sentence Length = the number of words / the number of sentences
df2['Average Sentence Length']=df2['WORD COUNT']/df2['Number_of_Sentences']
# Percentage of Complex words = the number of complex words / the number of words
df2['Percentage of Complex words']=df2['Complex_Word_Count']/df2['WORD COUNT']
#Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
df2['Fog Index']=0.4 * (df2['Average Sentence Length'] + df2['Percentage of Complex words'])
df2['AVG NUMBER OF WORDS PER SENTENCE']=df2['TOTAL NO OF WORDS']/df2['Number_of_Sentences']
df3=df2.drop(['full','TOTAL NO OF WORDS'],axis=1)
df3.head()


# In[289]:




# Assuming you have a DataFrame called 'df'

# Save the DataFrame to a CSV file
df3.to_csv('blackcoffer_assignment.csv', index=False)


# In[ ]:




