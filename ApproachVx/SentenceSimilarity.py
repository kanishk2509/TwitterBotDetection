
######################
# Loading word2vec
######################
import os
from threading import Semaphore
import gensim
from gensim.models import KeyedVectors

pathToBinVectors = '/Users/kanishksinha/Downloads/GoogleNews-vectors-negative300.bin'
newFilePath = '/Users/kanishksinha/Downloads/GoogleNews-vectors-negative300-normed.bin'

if os.path.isfile(newFilePath):
    print("File exists... please wait")
    model = KeyedVectors.load(newFilePath, mmap='r')
    model.syn0norm = model.syn0  # prevent recalc of normed vectors
    model.most_similar('stuff')  # any word will do: just to page all in
    Semaphore(0).acquire()  # just hang until process killed
else:
    print("Loading the data file... Please wait...")
    model = gensim.models.KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
    model.init_sims(replace=True)
    newFilePath = '/Users/kanishksinha/Downloads/GoogleNews-vectors-negative300-normed.bin'
    model.save(newFilePath)
    print("Successfully loaded 3.6 G bin file!")

# How to call one word vector?
# model1['resume'] -> This will return NumPy vector of the word "resume".

import numpy as np
import math
from scipy.spatial import distance

from random import sample
from nltk.corpus import stopwords


class PhraseVector:
    def __init__(self, phrase):
        self.vector = self.PhraseToVec(phrase)
    # <summary> Calculates similarity between two sets of vectors based on the averages of the sets.</summary>
    # <param>name = "vectorSet" description = "An array of arrays that needs to be condensed into a single array (vector). In this class, used to convert word vecs to phrases."</param>
    # <param>name = "ignore" description = "The vectors within the set that need to be ignored. If this is an empty list, nothing is ignored. In this class, this would be stop words."</param>
    # <returns> The condensed single vector that has the same dimensionality as the other vectors within the vecotSet.</returns>
    def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore = []):
        if len(ignore) == 0:
            return np.mean(vectorSet, axis = 0)
        else:
            return np.dot(np.transpose(vectorSet) ,ignore ) /sum(ignore)

    def PhraseToVec(self, phrase):
        cachedStopWords = stopwords.words("english")
        phrase = phrase.lower()
        wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
        vectorSet = []
        for aWord in wordsInPhrase:
            try:
                wordVector =model[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)

    # <summary> Calculates Cosine similarity between two phrase vectors.</summary>
    # <param> name = "otherPhraseVec" description = "The other vector relative to which similarity is to be calculated."</param>
    def CosineSimilarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / \
                    (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity


if __name__ == "__main__":
    print("###################################################################")
    print("###################################################################")
    print("########### WELCOME TO THE PHRASE SIMILARITY CALCULATOR ###########")
    print("###################################################################")
    print("###################################################################")

    text1 = 'Matt Lieber is a garment that the wind shook.'
    text2 = 'Matt Lieber is a final shrug of the shoulders.'

    phraseVector1 = PhraseVector(text1)
    phraseVector2 = PhraseVector(text2)
    similarityScore = phraseVector1.CosineSimilarity(phraseVector2.vector)

    print("###################################################################")
    print("Similarity Score: ", similarityScore)
    print("###################################################################")

