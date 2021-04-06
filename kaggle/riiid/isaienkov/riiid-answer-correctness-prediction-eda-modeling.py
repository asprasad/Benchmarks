#!/usr/bin/env python

# Source : https://www.kaggle.com/isaienkov/riiid-answer-correctness-prediction-eda-modeling/

# coding: utf-8

# <h1><center>Riiid! Answer Correctness Prediction. Data Analysis and visualization.</center></h1>
# 
# <center><img src="https://www.riiid.co/assets/opengraph.png"></center>
# 
# 
# 

# #### In this competition, your challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. You will pair your machine learning skills using Riiid’s EdNet data.
# 
# #### Let's check our data before we start this competition!

# <a id="top"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; border:0' role="tab" aria-controls="home"><center>Quick navigation</center></h3>
# 
# * [1. train.csv](#1)
# * [2. questions.csv](#2)
# * [3. lectures.csv](#3)
# * [4. example_test.csv](#4)
# * [5. Modeling](#5)

# In[1]:

import os
import numpy as np
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

import optuna
from optuna.samplers import TPESampler

# import riiideducation
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


scriptPath = os.path.realpath(__file__)
rootPath = os.path.dirname(scriptPath)
rootPath = os.path.dirname(rootPath)
dataDirPath = os.path.join(rootPath, "data")

# In[2]:


sampler = TPESampler(
    seed=666
)


# <a id="1"></a>
# <h2 style='background:black; border:0; color:white'><center>1. train.csv<center><h2>

# **train.csv**
# 
# * **row_id**: (int64) ID code for the row.
# 
# * **timestamp**: (int64) the time between this user interaction and the first event from that user.
# 
# * **user_id**: (int32) ID code for the user.
# 
# * **content_id**: (int16) ID code for the user interaction
# 
# * **content_type_id**: (int8) 0 if the event was a question being posed to the user, 1 if the event was the user watching a lecture.
# 
# * **task_container_id**: (int16) Id code for the batch of questions or lectures. For example, a user might see three questions in a row before seeing the explanations for any of them. Those three would all share a task_container_id. Monotonically increasing for each user.
# 
# * **user_answer**: (int8) the user's answer to the question, if any. Read -1 as null, for lectures.
# 
# * **answered_correctly**: (int8) if the user responded correctly. Read -1 as null, for lectures.
# 
# * **prior_question_elapsed_time**: (float32) How long it took a user to answer their previous question bundle, ignoring any lectures in between. The value is shared across a single question bundle, and is null for a user's first question bundle or lecture. Note that the time is the total time a user took to solve all the questions in the previous bundle.
# 
# * **prior_question_had_explanation**: (bool) Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle, ignoring any lectures in between. The value is shared across a single question bundle, and is null for a user's first question bundle or lecture. Typically the first several questions a user sees were part of an onboarding diagnostic test where they did not get any feedback.

# In[3]:


types = {
        'row_id': 'int64', 
        'timestamp': 'int64', 
        'user_id': 'int32', 
        'content_id': 'int16', 
        'content_type_id': 'int8',
        'task_container_id': 'int16', 
        'user_answer': 'int8', 
        'answered_correctly': 'int8', 
        'prior_question_elapsed_time': 'float32', 
        'prior_question_had_explanation': 'boolean'
}


# In[4]:


train_df = pd.read_csv(
    os.path.join(dataDirPath, 'train.csv'), #'/kaggle/input/riiid-test-answer-prediction/train.csv', 
    low_memory=False, 
    nrows=10**6, 
    dtype=types
)

train_df.head()


# In[5]:


print('Part of missing values for every column')
print(train_df.isnull().sum() / len(train_df))


# In[6]:


WIDTH = 800


# In[7]:


ds = train_df['user_id'].value_counts().reset_index()

ds.columns = [
    'user_id', 
    'count'
]

ds['user_id'] = ds['user_id'].astype(str) + '-'
ds = ds.sort_values(['count']).tail(40)

fig = px.bar(
    ds, 
    x='count', 
    y='user_id', 
    orientation='h', 
    title='Top 40 users by number of actions', 
    width=WIDTH,
    height=900 
)

#fig.show()


# In[8]:


ds = train_df['user_id'].value_counts().reset_index()

ds.columns = [
    'user_id', 
    'count'
]

ds = ds.sort_values('user_id')

fig = px.line(
    ds, 
    x='user_id', 
    y='count', 
    title='User action distribution', 
    width=WIDTH,
    height=600 
)

#fig.show()


# In[9]:


ds = train_df['content_id'].value_counts().reset_index()

ds.columns = [
    'content_id', 
    'count'
]

ds['content_id'] = ds['content_id'].astype(str) + '-'
ds = ds.sort_values(['count']).tail(40)

fig = px.bar(
    ds, 
    x='count', 
    y='content_id', 
    orientation='h', 
    title='Top 40 most useful content_ids',  
    width=WIDTH,
    height=900
)

#fig.show()


# In[10]:


ds = train_df['content_id'].value_counts().reset_index()

ds.columns = [
    'content_id', 
    'count'
]

ds = ds.sort_values('content_id')

fig = px.line(
    ds, 
    x='content_id', 
    y='count', 
    title='content_id action distribution', 
    width=WIDTH,
    height=600 
)

#fig.show()


# In[11]:


ds = train_df['content_type_id'].value_counts().reset_index()

ds.columns = [
    'content_type_id', 
    'percent'
]

ds['percent'] /= len(train_df)

fig = px.pie(
    ds, 
    names='content_type_id', 
    values='percent', 
    title='Lecures & questions', 
    width=WIDTH,
    height=500 
)

#fig.show()


# In[12]:


ds = train_df['task_container_id'].value_counts().reset_index()

ds.columns = [
    'task_container_id', 
    'count'
]

ds['task_container_id'] = ds['task_container_id'].astype(str) + '-'
ds = ds.sort_values(['count']).tail(40)

fig = px.bar(
    ds, 
    x='count', 
    y='task_container_id', 
    orientation='h', 
    title='Top 40 most useful task_container_ids', 
    width=WIDTH,
    height=900
)

#fig.show()


# In[13]:


ds = train_df['task_container_id'].value_counts().reset_index()

ds.columns = [
    'task_container_id', 
    'count'
]

ds = ds.sort_values('task_container_id')

fig = px.line(
    ds, 
    x='task_container_id', 
    y='count', 
    title='task_container_id action distribution', 
    width=WIDTH,
    height=600  
)

#fig.show()


# In[14]:


ds = train_df['user_answer'].value_counts().reset_index()

ds.columns = [
    'user_answer', 
    'percent_of_answers'
]

ds['percent_of_answers'] /= len(train_df)
ds = ds.sort_values(['percent_of_answers'])

fig = px.bar(
    ds, 
    x='user_answer', 
    y='percent_of_answers', 
    orientation='v', 
    title='Percent of user answers for every option', 
    width=WIDTH,
    height=400 
)

#fig.show()


# In[15]:


ds = train_df['answered_correctly'].value_counts().reset_index()

ds.columns = [
    'answered_correctly', 
    'percent_of_answers'
]

ds['percent_of_answers'] /= len(train_df)
ds = ds.sort_values(['percent_of_answers'])

fig = px.pie(
    ds, 
    names='answered_correctly', 
    values='percent_of_answers', 
    title='Percent of correct answers', 
    width=WIDTH,
    height=500 
)

#fig.show()


# In[16]:


fig = make_subplots(rows=3, cols=2)

traces = [
    go.Bar(
        x=[
            -1, 0, 1
        ], 
        y=[
            len(train_df[(train_df['user_answer'] == item) & (train_df['answered_correctly'] == -1)]),
            len(train_df[(train_df['user_answer'] == item) & (train_df['answered_correctly'] == 0)]),
            len(train_df[(train_df['user_answer'] == item) & (train_df['answered_correctly'] == 1)])
        ], 
        name='Option: ' + str(item),
        text = [
            str(round(100 * len(train_df[(train_df['user_answer'] == item) & (train_df['answered_correctly'] == -1)]) / len(train_df[(train_df['user_answer'] == item)]), 2)) + '%',
            str(round(100 * len(train_df[(train_df['user_answer'] == item) & (train_df['answered_correctly'] == -0)]) / len(train_df[(train_df['user_answer'] == item)]), 2)) + '%',
            str(round(100 * len(train_df[(train_df['user_answer'] == item) & (train_df['answered_correctly'] == 1)]) / len(train_df[(train_df['user_answer'] == item)]), 2)) + '%',
        ],
        textposition='auto'
    ) for item in train_df['user_answer'].unique().tolist()
]

for i in range(len(traces)):
    fig.append_trace(
        traces[i], 
        (i // 2) + 1, 
        (i % 2)  + 1
    )

fig.update_layout(
    title_text='Percent of correct answers for every option',
    height=900,
    width=WIDTH
)

#fig.show()


# In[17]:


fig = px.histogram(
    train_df, 
    x="prior_question_elapsed_time",
    nbins=50,
    title='prior_question_elapsed_time distribution',
    width=WIDTH,
    height=500
)

#fig.show()


# <a id="2"></a>
# <h2 style='background:black; border:0; color:white'><center>2. questions.csv<center><h2>

# **questions.csv**: metadata for the questions posed to users.
# 
# * **question_id**: foreign key for the train/test content_id column, when the content type is question (0).
# 
# * **bundle_id**: code for which questions are served together.
# 
# * **correct_answer**: the answer to the question. Can be compared with the train user_answer column to check if the user was right.
# 
# * **part**: top level category code for the question.
# 
# * **tags**: one or more detailed tag codes for the question. The meaning of the tags will not be provided, but these codes are sufficient for clustering the questions together.

# In[18]:


questions = pd.read_csv(os.path.join(dataDirPath, 'questions.csv')) #pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')
questions.head()


# In[19]:


print('Part of missing values for every column')
print(questions.isnull().sum() / len(questions))


# In[20]:


questions['tag'] = questions['tags'].str.split(' ')
questions = questions.explode('tag')
questions = pd.merge(
    questions, 
    questions.groupby('question_id')['tag'].count().reset_index(), 
    on='question_id'
)

questions = questions.drop(['tag_x'], axis=1)

questions.columns = [
    'question_id', 
    'bundle_id', 
    'correct_answer', 
    'part', 
    'tags', 
    'tags_number'
]

questions = questions.drop_duplicates()


# In[21]:


ds = questions['correct_answer'].value_counts().reset_index()

ds.columns = [
    'correct_answer', 
    'number_of_answers'
]

ds['correct_answer'] = ds['correct_answer'].astype(str) + '-'
ds = ds.sort_values(['number_of_answers'])

fig = px.bar(
    ds, 
    x='number_of_answers', 
    y='correct_answer', 
    orientation='h', 
    title='Number of correct answers per group', 
    width=WIDTH,
    height=300
)

#fig.show()


# In[22]:


ds = questions['part'].value_counts().reset_index()

ds.columns = [
    'part', 
    'count'
]

ds['part'] = ds['part'].astype(str) + '-'
ds = ds.sort_values(['count'])

fig = px.bar(
    ds, 
    x='count', 
    y='part', 
    orientation='h', 
    title='Parts distribution',
    width=WIDTH,
    height=400
)

#fig.show()


# In[23]:


ds = questions['tags_number'].value_counts().reset_index()

ds.columns = [
    'tags_number', 
    'count'
]

ds['tags_number'] = ds['tags_number'].astype(str) + '-'
ds = ds.sort_values(['tags_number'])

fig = px.bar(
    ds, 
    x='count', 
    y='tags_number', 
    orientation='h', 
    title='Number tags distribution', 
    width=WIDTH,
    height=400 
)

#fig.show()


# In[24]:


check = pd.DataFrame(questions['tags'].str.split(' ')).explode('tags').reset_index()
check = check['tags'].value_counts().reset_index()

check.columns = [
    'tag', 
    'count'
]

check['tag'] = check['tag'].astype(str) + '-'
check = check.sort_values(['count']).tail(40)

fig = px.bar(
    check, 
    x='count', 
    y='tag', 
    orientation='h', 
    title='Top 40 most useful tags', 
    width=WIDTH,
    height=900 
)

#fig.show()


# <a id="3"></a>
# <h2 style='background:black; border:0; color:white'><center>3. lectures.csv<center><h2>

# **lectures.csv**: metadata for the lectures watched by users as they progress in their education.
# 
# * **lecture_id**: foreign key for the train/test content_id column, when the content type is lecture (1).
# 
# * **part**: top level category code for the lecture.
# 
# * **tag**: one tag codes for the lecture. The meaning of the tags will not be provided, but these codes are sufficient for clustering the lectures together.
# 
# * **type_of**: brief description of the core purpose of the lecture

# In[25]:


lectures = pd.read_csv(os.path.join(dataDirPath, 'lectures.csv')) #'/kaggle/input/riiid-test-answer-prediction/lectures.csv')
lectures.head()


# In[26]:


print('Part of missing values for every column')
print(lectures.isnull().sum() / len(lectures))


# In[27]:


ds = lectures['tag'].value_counts().reset_index()

ds.columns = [
    'tag', 
    'count'
]

ds['tag'] = ds['tag'].astype(str) + '-'
ds = ds.sort_values(['count']).tail(40)

fig = px.bar(
    ds, 
    x='count', 
    y='tag', 
    orientation='h', 
    title='Top 40 lectures by number of tags', 
    height=900, 
    width=WIDTH
)

#fig.show()


# In[28]:


ds = lectures['part'].value_counts().reset_index()

ds.columns = [
    'part', 
    'count'
]

ds['part'] = ds['part'].astype(str) + '-'
ds = ds.sort_values(['count'])

fig = px.bar(
    ds, 
    x='count', 
    y='part', 
    orientation='h', 
    title='Parts distribution', 
    height=400, 
    width=WIDTH
)

#fig.show()


# In[29]:


ds = lectures['type_of'].value_counts().reset_index()

ds.columns = [
    'type_of', 
    'count'
]

ds = ds.sort_values(['count'])

fig = px.bar(
    ds, 
    x='count', 
    y='type_of', 
    orientation='h', 
    title='type_of column distribution', 
    height=300, 
    width=WIDTH
)

#fig.show()


# <a id="4"></a>
# <h2 style='background:black; border:0; color:white'><center>4. example_test.csv<center><h2>

# **example_test.csv** Three sample groups of the test set data as it will be delivered by the time-series API. The format is largely the same as train.csv. There are two different rows that mirror what information the AI tutor actually has available at any given time, but with the user interactions grouped together for the sake of API performance rather than strictly showing information for a single user at a time. Some questions will appear in the hidden test set that have NOT been presented in the train set, emulating the challenge of quickly adapting to modeling newly introduced questions. Their metadata is still in question.csv as usual.
# 
# prior_group_responses (string) provides all of the user_answer entries for previous group in a string representation of a list in the first row of the group. All other rows in each group are null. If you are using Python, you will likely want to call eval on the non-null rows. Some rows may be null, or empty lists.
# 
# prior_group_answers_correct (string) provides all the answered_correctly field for previous group, with the same format and caveats as prior_group_responses. Some rows may be null, or empty lists.

# In[30]:


test_ex = pd.read_csv(os.path.join(dataDirPath, 'example_test.csv'))#'/kaggle/input/riiid-test-answer-prediction/example_test.csv')
test_ex


# <a id="5"></a>
# <h2 style='background:black; border:0; color:white'><center>5. Modeling<center><h2>

# In[31]:


used_data_types_dict = {
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float16',
    'prior_question_had_explanation': 'boolean'
}


# In[32]:


train_df = pd.read_csv(
    os.path.join(dataDirPath, 'train.csv'), #'/kaggle/input/riiid-test-answer-prediction/train.csv',
    usecols = used_data_types_dict.keys(),
    dtype=used_data_types_dict, 
    index_col = 0,
    nrows=10**7
)


# In[33]:


features_df = train_df.iloc[:int(9/10 * len(train_df))]
train_df = train_df.iloc[int(9/10 * len(train_df)):]


# In[34]:


train_questions_only_df = features_df[features_df['answered_correctly']!=-1]
grouped_by_user_df = train_questions_only_df.groupby('user_id')

user_answers_df = grouped_by_user_df.agg(
    {
        'answered_correctly': [
            'mean', 
            'count', 
            'std', 
            'median', 
            'skew'
        ]
    }
).copy()

user_answers_df.columns = [
    'mean_user_accuracy', 
    'questions_answered', 
    'std_user_accuracy', 
    'median_user_accuracy', 
    'skew_user_accuracy'
]

user_answers_df


# In[35]:


grouped_by_content_df = train_questions_only_df.groupby('content_id')
content_answers_df = grouped_by_content_df.agg(
    {
        'answered_correctly': [
            'mean', 
            'count', 
            'std', 
            'median', 
            'skew'
        ]
    }
).copy()

content_answers_df.columns = [
    'mean_accuracy', 
    'question_asked', 
    'std_accuracy', 
    'median_accuracy', 
    'skew_accuracy'
]

content_answers_df


# In[36]:


del features_df
del grouped_by_user_df
del grouped_by_content_df


# In[37]:


features = [
    'mean_user_accuracy', 
    'questions_answered',
    'std_user_accuracy', 
    'median_user_accuracy',
    'skew_user_accuracy',
    'mean_accuracy', 
    'question_asked',
    'std_accuracy', 
    'median_accuracy',
    'prior_question_elapsed_time', 
    'prior_question_had_explanation',
    'skew_accuracy'
]

target = 'answered_correctly'


# In[38]:


train_df = train_df[train_df[target] != -1]


# In[39]:


train_df = train_df.merge(user_answers_df, how='left', on='user_id')
train_df = train_df.merge(content_answers_df, how='left', on='content_id')


# In[40]:


train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].fillna(value=False).astype(bool)
train_df = train_df.fillna(value=0.5)


# In[41]:


train_df = train_df[features + [target]]
train_df = train_df.replace([np.inf, -np.inf], np.nan)
train_df = train_df.fillna(0.5)

train_df


# In[42]:


train_df, test_df = train_test_split(train_df, random_state=666, test_size=0.2)


# In[43]:


rfe = RFE(
    estimator=DecisionTreeClassifier(
        random_state=666
    ), 
    n_features_to_select=8
)


# In[44]:


rfe.fit(train_df[features], train_df[target])
X_transformed = rfe.transform(train_df[features])
X_transformed = pd.DataFrame(X_transformed)


# In[45]:


X_transformed.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8']
X_transformed


# In[46]:


X_transformed_test = rfe.transform(test_df[features])
X_transformed_test = pd.DataFrame(X_transformed_test)


# In[47]:


X_transformed_test.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8']
X_transformed_test


# In[48]:


X_transformed_test['col_1'] = X_transformed_test['col_1'].astype(np.float32)
X_transformed_test['col_2'] = X_transformed_test['col_2'].astype(np.float32)
X_transformed_test['col_3'] = X_transformed_test['col_3'].astype(np.int32)
X_transformed_test['col_4'] = X_transformed_test['col_4'].astype(np.float32)
X_transformed_test['col_5'] = X_transformed_test['col_5'].astype(np.int32)
X_transformed_test['col_6'] = X_transformed_test['col_6'].astype(np.int32)
X_transformed_test['col_7'] = X_transformed_test['col_7'].astype(np.int32)
X_transformed_test['col_8'] = X_transformed_test['col_8'].astype(np.float32)

X_transformed['col_1'] = X_transformed['col_1'].astype(np.float32)
X_transformed['col_2'] = X_transformed['col_2'].astype(np.float32)
X_transformed['col_3'] = X_transformed['col_3'].astype(np.int32)
X_transformed['col_4'] = X_transformed['col_4'].astype(np.float32)
X_transformed['col_5'] = X_transformed['col_5'].astype(np.int32)
X_transformed['col_6'] = X_transformed['col_6'].astype(np.int32)
X_transformed['col_7'] = X_transformed['col_7'].astype(np.int32)
X_transformed['col_8'] = X_transformed['col_8'].astype(np.float32)


# In[49]:


def create_model(trial):
   num_leaves = trial.suggest_int("num_leaves", 2, 31)
   n_estimators = trial.suggest_int("n_estimators", 50, 300)
   max_depth = trial.suggest_int('max_depth', 3, 8)
   min_child_samples = trial.suggest_int('min_child_samples', 100, 1200)
   learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)
   min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 90)
   bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.0001, 1.0)
   feature_fraction = trial.suggest_uniform('feature_fraction', 0.0001, 1.0)
   
   model = LGBMClassifier(
       num_leaves=num_leaves,
       n_estimators=n_estimators, 
       max_depth=max_depth, 
       min_child_samples=min_child_samples, 
       min_data_in_leaf=min_data_in_leaf,
       learning_rate=learning_rate,
       feature_fraction=feature_fraction,
       random_state=666
   )
   return model

def objective(trial):
   model = create_model(trial)
   model.fit(X_transformed, train_df[target])
   score = roc_auc_score(
       test_df[target].values, 
       model.predict_proba(X_transformed_test)[:,1]
   )
   return score

# uncomment to use optuna
# final params is in study.best_params
# study = optuna.create_study(direction="maximize", sampler=sampler)
# study.optimize(objective, n_trials=70)
# params = study.best_params
# params['random_state'] = 666



params = {
   'bagging_fraction': 0.5817242323514327,
   'feature_fraction': 0.6884588361650144,
   'learning_rate': 0.42887924851375825, 
   'max_depth': 6,
   'min_child_samples': 946, 
   'min_data_in_leaf': 47, 
   'n_estimators': 169,
   'num_leaves': 29,
   'random_state': 666
}

model = LGBMClassifier(
   **params
)

model.fit(X_transformed, train_df[target])
print('LGB score: ', roc_auc_score(test_df[target].values, model.predict_proba(X_transformed_test)[:,1]))


# In[50]:


# env = riiideducation.make_env()
# iter_test = env.iter_test()


# In[51]:


# for (test_df, sample_prediction_df) in iter_test:
#     test_df = test_df.merge(user_answers_df, how='left', on='user_id')
#     test_df = test_df.merge(content_answers_df, how='left', on='content_id')
#     test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value=False).astype(bool)
#     test_df.fillna(value = 0.5, inplace = True)
    
#     test = rfe.transform(test_df)
#     test = pd.DataFrame(test)
#     test['col_1'] = test['col_1'].astype(np.float32)
#     test['col_2'] = test['col_2'].astype(np.float32)
#     test['col_3'] = test['col_3'].astype(np.int32)
#     test['col_4'] = test['col_4'].astype(np.float32)
#     test['col_5'] = test['col_5'].astype(np.int32)
#     test['col_6'] = test['col_6'].astype(np.int32)
#     test['col_7'] = test['col_7'].astype(np.int32)
#     test['col_8'] = test['col_8'].astype(np.float32)


#     test_df['answered_correctly'] = model.predict_proba(test)[:,1]
#     env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])


# In[ ]:
# Inference

test_df_full = pd.read_csv(
    os.path.join(dataDirPath, 'train.csv'), #'/kaggle/input/riiid-test-answer-prediction/train.csv',
    usecols = used_data_types_dict.keys(),
    dtype=used_data_types_dict, 
    index_col = 0,
    nrows=10**7
)

for start in range(0, 10**7, 100): #(test_df, sample_prediction_df) in iter_test:
    test_df = test_df_full[start:start+100]
    test_df = test_df.merge(user_answers_df, how='left', on='user_id')
    test_df = test_df.merge(content_answers_df, how='left', on='content_id')
    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value=False).astype(bool)
    test_df.fillna(value = 0.5, inplace = True)

    test_df = test_df[features]
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.fillna(0.5)
    
    test = rfe.transform(test_df)
    test = pd.DataFrame(test)
    test.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8']
    test['col_1'] = test['col_1'].astype(np.float32)
    test['col_2'] = test['col_2'].astype(np.float32)
    test['col_3'] = test['col_3'].astype(np.int32)
    test['col_4'] = test['col_4'].astype(np.float32)
    test['col_5'] = test['col_5'].astype(np.int32)
    test['col_6'] = test['col_6'].astype(np.int32)
    test['col_7'] = test['col_7'].astype(np.int32)
    test['col_8'] = test['col_8'].astype(np.float32)

    test_df['answered_correctly'] = model.predict_proba(test)[:,1]
    # env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])


