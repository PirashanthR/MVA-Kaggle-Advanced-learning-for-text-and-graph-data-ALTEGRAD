

This repository contains the handing for the MVA master's class:
Advanced learning for text and graph data ALTEGRAD.
http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/contenus-/advanced-learning-for-text-and-graph-data-239506.kjsp?RH=1242430202531

The final project of this class was an inclass Kaggle Challenge.
The data of the challenge are provided in the train.csv and test.csv file.

The goal was to compare pairs of questions and detect duplicates.
Here is the link to the challenge page:
https://www.kaggle.com/c/altegrad-challenge-fall-17
This work has been done in a team of two: 
RATNAMOGAN Pirashanth 
SAYEM Othmane

Kaggle Team Name: Ratnamogan - Sayem

Using this implementation we have been ranked 4th among 52 teams (team up to 4 people) on both the public and the private leaderboard.


The code has been written in python 3.5.
The following librairies are needed in order to run it:


numpy
pandas
pickle
os
collections
random 
gensim
keras
xgboost
lightgbm
sklearn
scipy
math
nltk
networkx
igraph
fuzzywuzzy

The code has been generated in order to be quickly understandable and easy to use and modify by the whole team.
It's not Optimal at all (stemming is computed each time when it's needed for instance), but the goal wasn't to provide an optimal
code. 

In order to run the code and generate the needed submission file, one has to run the "main.py" file.
We have created 220 various features using various methods from different domain: NLP, graph, ...
We have used various ensemble methods to provide a good regularized outcome.
All the features are generated in the functions described in the folder "Features".
Preprocessing is described in the folder "Preprocessing".

The folder "unused ideas draft" contains the ideas that we have tried but that doesn't allow to improve our
outcome. (See report for details)



