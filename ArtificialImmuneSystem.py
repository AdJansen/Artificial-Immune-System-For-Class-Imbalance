import pandas as pd;
import random;
from statistics import fmean, stdev;

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import DistanceMetric
from sklearn.ensemble import RandomForestClassifier

class ArtificialImmuneSystem():
    
        
    #TODO Gaussian generation can be optimized by altering how the loops work, putting the col loop on the outside and pre-generated fmean, stdev for the col
            #Currently it generates those values for every antibody
            

    def get_bounds(self, minorityDF) -> dict:
        out = {}
        for col in minorityDF:
            colMax = minorityDF[col].max()
            colMin = minorityDF[col].min()
            out[col] = (colMin, colMax)
        return out


    ####### Creation ################
    # minorityDF - dataframe containing the minority class
    # totalPopulation - The total number of antibodies to create
    # weightingFunction - Can choose between uniform, triangular, ...
    # mode - for use with a triangular function - set to the percentage of the range you wish to be most represented (between 0.0 and 1.0)
    def Creation(self, minorityDF, totalPopulation : int, binaryColumns : list, weightingFunction : str = "uniform", mode : float = 0.5): 
        
        if(minorityDF.isnull().values.any()):
            raise ValueError("Minority Class DataFrame contains NaN")
        
        population = [] #Initializing the empty population
        if mode < 0.0 or mode > 1:
            raise Exception("mode must be between float value between 0.0 and 1.0")
        
        if weightingFunction not in ('uniform', 'triangular', 'gaussian'):
            raise Exception("Unknown function chosen, please use one of 'uniform', 'triangular', or 'gaussian'")

        bounds = self.get_bounds(minorityDF)
        
        if weightingFunction in ["uniform", "triangular"]: #If Generating via uniform or triangular distribution, loop through bounds of columns
            for i in range(totalPopulation): #For every antibody to be created

                antibody = [] #Initializing a single antibody
                for key,bnd in bounds.items(): #Iterate through the columns/dimensions/features of the minority class for each antibody 
                    if key in binaryColumns:
                        antibody += [random.randint(int(bnd[0]),int(bnd[1]))]
                    else:
                        if weightingFunction == "uniform":
                            antibody += [round(random.uniform(bnd[0],bnd[1]),4)] #Add a random value between the lower and upper bounds to the antibody

                        elif (weightingFunction == "triangular"):
                            
                            tri_tip = ( ((bnd[1]-bnd[0]) * mode) + bnd[0] ) #multiplying the difference by the percentage, plus the low bound gives us the point between the two, but percentile

                            if tri_tip < bnd[0]: #Error checks to make sure that the emphasized point isn't outside the bounds
                                tri_tip = bnd[0]
                            elif tri_tip > bnd[1]:
                                tri_tip = bnd[1]

                            antibody += [round( random.triangular(bnd[0],bnd[1], tri_tip), 5)]

                population+=[antibody] #add the created antibody to the population

        elif (weightingFunction == 'gaussian'): #If Generating via Gaussian, loop through columns of dataframe

            for i in range(totalPopulation): #For every antibody to be created

                antibody = [] #Initializing a single antibody
                for bnd in minorityDF: #Iterate over columns in the dataframe
                    values = minorityDF[bnd].tolist() #convert series to list
                    if bnd in binaryColumns:
                        antibody += [random.randint(bounds[bnd][0],bounds[bnd][1])]
                    else:
                        antibody += [round(random.gauss(fmean(values) , stdev(values)), 5)] #using median and stdeviation of values, radomize over gauss

            
                population+=[antibody] #add the created antibody to the population

                
        popDF = pd.DataFrame(population, columns = minorityDF.columns.values)
        return popDF, bounds


    ####### Fitness ################
    def fitness(self, model, feat, label, iterations, scorer):
        #scorer is the name of the function wee aree using to evaluate our dataset
        #it should be a function with signature scorer(model, feature, label) which should return only a single value.
        return cross_val_score(model, feat, label, cv = iterations, scoring = scorer)

    def distance(self, x, y, metric):
        
        #get the distance between two sets of data x and y, they should be the same size
        #metric is the string metric to be used to measure distance

        dist = DistanceMetric.get_metric(metric)
        return dist.pairwise(x,y)


    ####### Mutation ################
    def mutatePopulation (self, antiPopulation, bounds, binaryColumns : list):
        #antiPopulation is the population of antibodies to be mutated
        #bounds is a dictionary of the bounds of each column in the population
        #binaryColumns is a list of the columns that are binary
        #returns a new mutated population of antibodies
        antiPopulation = antiPopulation.copy()
        for col in antiPopulation:
            if col in binaryColumns: #Binary Columns must be handled differently than continuous
                
                antiPopulation[col] = antiPopulation[col].map(lambda x : (random.randint(0,1)))
            else:
                bnd_range = bounds[col][1] - bounds[col][0] #total range of bounds is high - low

                #Setting the low and high bounds to be centered around 0
                hi_bnd = bnd_range/2 
                low_bnd = (0-bnd_range/2)

                #print("Low bound around 0 = " + str(low_bnd) +"| Hi bnd around 0 = "+ str(hi_bnd))
                #print(round(random.uniform(low_bnd,hi_bnd),4))

                antiPopulation[col] = antiPopulation[col].map(lambda x : x+round(random.uniform(low_bnd,hi_bnd),4))
            
        return antiPopulation


    def comparePopulationsDepreciated(self, population1, population2, labels1, labels2, estimator, iterations, scorer):

        score1 = fmean(self.fitness(estimator, population1, labels1.values.ravel(), iterations, scorer))
        score2 = fmean(self.fitness(estimator, population2, labels2.values.ravel(), iterations, scorer))

        if score1 > score2:
            winning_population = population1
            winning_labels = labels1
        else:
            winning_population = population2
            winning_labels = labels2

        for col in winning_labels:
            winning_population = winning_population.join(winning_labels[col])

        return winning_population



    def comparePopulations(self,population1, population2, labels1, labels2, estimator, iterations, scorer):
        score1 = fmean(self.fitness(estimator, population1, labels1.values.ravel(), iterations, scorer))
        score2 = fmean(self.fitness(estimator, population2, labels2.values.ravel(), iterations, scorer))

        if abs(score1 - score2) < 0.005:
            return False
        elif (score1>score2):
            return False
        else:
            return True

    #TODO : add parameter that defines which column is the label
    #separate a df into features and labels
    def separate_df(self, df):

        columns = df.columns.to_list()
        columns_drop = columns.pop(-1)

        labels = df.drop(columns, axis=1)
        features = df.drop(columns_drop, axis=1)

        return features, labels

    def AIS(self,df,max_rounds, totalPopulation):

        #change hardcoded
        #should be the minority df instead
        
        initial_population, bounds = self.Creation(df,totalPopulation,['5'], weightingFunction='uniform')
        
        antibody_population = self.mutatePopulation(initial_population,bounds,['5'])
        
        count = 0
        no_change = 0

        current_gen, current_labels = self.separate_df(initial_population)
        next_gen, next_labels = self.separate_df(antibody_population)
    
        while( (count<max_rounds) and (no_change < 5) ):
            count+=1

            #change hardcoded
            if(self.comparePopulations(current_gen,next_gen,current_labels,next_labels,svm.SVC(random_state=0), 5, 'f1_macro')):
                
                no_change = 0

                current_gen = next_gen.copy()
                current_labels = next_labels.copy()

                current = current_gen.copy()
                for col in current_labels:
                    current = current.join(current_labels[col])
                #current.append(current_labels)

                #need to update bounds
                antibody_population = self.mutatePopulation(current,bounds,['5'])
                next_gen, next_labels = self.separate_df(antibody_population)

                # print('obama')
                # print(current_labels)
                # print(next_labels)
                

            else:

                no_change+=1
                # will this give the same thing every time?
                current = current_gen.copy()
                for col in current_labels:
                    current = current.join(current_labels[col])
                #current.co(current_labels)
                
                antibody_population = self.mutatePopulation(current,bounds,['5'])
                next_gen, next_labels = self.separate_df(antibody_population)
                current_gen, current_labels = self.separate_df(current)
                
                # print('trump')
                # print(current_labels)
                # print(next_labels)
                # print(current_gen)
                # print(current)
                # print(antibody_population)
                


        return current_gen, count