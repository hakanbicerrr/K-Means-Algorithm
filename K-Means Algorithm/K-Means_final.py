import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io
import copy
import random
#This is for being able to control results easily.Because this provides fixed random numbers every program.
np.random.seed(200)

def main():
    #Create data which contains coordinates in plane
    #mat = [[12, 39], [20, 36], [28, 30], [18, 52], [29, 54], [33, 46], [24, 55], [45, 59], [45, 63], [52, 70],
     #      [51, 66], [52, 63], [55, 58], [53, 23], [55, 14], [61, 8], [64, 19], [69, 7], [72, 24]]
    mat = scipy.io.loadmat('clusteringData.mat')
    mat = np.asarray(mat["X"])
    mat = mat.transpose()
    #Create empty list for x and y coordinates

    #z = [[0]*len(mat)]*len(mat[0])
    z = [[] for i in range(len(mat[0]))]
    a = 0
    for i in range(len(mat[0])):
        for j in range(len(mat)):
            z[i].append(mat[j][i])
    #Create list containing x and y coordinates seperately


    #Define how many cluster desired
    k = int(input("Please enter desired number of cluster: "))
    #Define initial centroids.This can be random or anything.
    centroids = create_initial_centroids(mat,z,k)
    centroids = centroids.astype("float")
    print("Initial centroids:\n",centroids)

    iteration = 5

    label = create_cluster_label(z,centroids,iteration,k)

def create_cluster_label(z,centroids,iteration,k):



    for iter in range(iteration):
        cluster_label = []
        label = []
        for i in range(len(z[0])):
            distance = {}
            a = [row[i] for row in z]
            for j in range(k):

                distance[j] = compute_oklidean_distance(a,centroids[j])
            label = create_label(a,distance,centroids)

            #new_centroid = calculate_new_centroid(label[1],label[2])
            #centroids[label[0]] = new_centroid

            cluster_label.append(label)

        avg = []
        avg_final = []
        for dim in range(k):
            avg_final = []
            for i in range(len(cluster_label)):
                if cluster_label[i][0] == dim:
                    #for j in range(len(a)):
                    avg.append(cluster_label[i][1])
                    #avg_y.append(cluster_label[i][2])
            a = []
            for ite in range(len(avg[0])):
                for it in range(len(avg)):
                    a.append(avg[it][ite])
                avg_final.append(np.mean(a))
                a = []

            #avg = [[] for i in range(len(a))]
            avg = []
        #print(avg_x,avg_y)
            centroids[dim] = avg_final
    s0 = s1 = s2 = 0
    for i in range(len(cluster_label)):
        if cluster_label[i][0] == 0:
            s0 += 1
        elif cluster_label[i][0] == 1:
            s1 += 1
        elif cluster_label[i][0] == 2:
            s2 += 1

    print("New Centroids:\n",centroids)
    labeler = []
    for i in range(len(cluster_label)):
        labeler.append(cluster_label[i][0])
    print("Cluster Label: ", cluster_label)
    print(labeler)
    #print(cluster_label[0])
    print(s0, s1, s2)
def calculate_new_centroid(a,centroid):
    new_centroid = []
    for i in range(len(a)):
        new_centroid.append((a[i] + centroid[i]) / 2)
    return new_centroid

def create_label(a,distance,centroids):
    #Get index of minimum distance as cluster number
    distance_min_index = min(distance,key=distance.get)
    return [distance_min_index,a,centroids[distance_min_index]]


def compute_oklidean_distance(z,centroid):
    dist = 0
    for i in range(len(z)):
        dist += (z[i]-centroid[i])**2
    dist = np.sqrt(dist)
    return dist

def create_initial_centroids(data,z,k):
    #Declare centroids
    centroids = [[] for i in range(k)]
    a=0
    #temp_list = copy.deepcopy(data)
    #Initialize and assign values randomly to centroids
    for i in range(k):
        for j in range(len(z)):
            centroids[i].append(z[j][a+4])
        a+=1
        ############################################################
        #index = random.randrange(len(temp_list))
        #centroids[i].append(temp_list[index])

    #$print(centroids)
    #centroids = np.asarray(centroids)
    #centroids = centroids.flatten()
    #centroids = np.reshape(centroids,(3,5))
    ################################################################
    #Convert list to array to be able to do math easily
    return np.array(centroids)





















if __name__ == "__main__":

    main()