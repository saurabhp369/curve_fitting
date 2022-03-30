import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def linear_ls(x,y):
    x = np.vstack((x,np.ones(x.shape)))
    x = np.transpose(x)
    p = np.linalg.inv(np.matmul(np.transpose(x),x))
    q = np.matmul(np.transpose(x),y)
    b = np.matmul(p,q)
    return b

def total_ls(x):
    a,b = np.linalg.eigh(x)
    return b[:,0]

def ransac(x,y):
    max_inlier_count = 0
    B_model = np.zeros(2)
    thresh_dist =1000
    N = int(math.log(1-0.95)/math.log(1-math.pow((1-0.5),2)))
    i = 0
   
    while i<N:
        
        # selecting two points at random from the dataset
        index = np.random.choice(x.shape[0], 2)
        p_x = x[index]
        p_y = y[index]
        if((p_x[0] == p_x[1]) or (p_y[0] == p_y[1])):
            continue
        # fitting a line through these two points
        B_current = np.zeros(2)
        B_current[0] = (p_y[1] - p_y[0])/(p_x[1]-p_x[0])
        B_current[1] = p_y[1] - B_current[0]*p_x[1]

        d = np.abs(y - B_current[0]*x - B_current[1]*np.ones(x.shape))
        inliers = np.where(d<thresh_dist)
        inlier_count = len(inliers[0])
        
        if inlier_count>max_inlier_count:
            B_model = B_current
            max_inlier_count = inlier_count
        e = 1-(max_inlier_count/x.shape[0])
        N = int(math.log(1-0.95)/math.log(1-math.pow((1-e),2)))   

        i+=1
    return B_model

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv (base_dir + 'data.csv', usecols=['age', 'charges'])
    data = df.to_numpy(dtype='float')
    age = data[:,0]
    cost = data[:,1]
    n = age.shape[0]
    mean_age = np.mean(age)
    mean_cost = np.mean(cost)
    var_age = np.matmul((age-mean_age), np.transpose(age-mean_age))/n
    var_cost = np.matmul((cost-mean_cost), np.transpose(cost-mean_cost))/n
    var_age_cost = np.matmul((age-mean_age), np.transpose(cost-mean_cost))/n
    cov_mat = np.array([[var_age, var_age_cost],[var_age_cost, var_cost]])
    e_values, e_vect = np.linalg.eig(cov_mat)
    origin = [mean_age,mean_cost]
    eig_vect1 = e_vect[:,0]
    eig_vect2 = e_vect[:,1]
    plt.scatter(age,cost, c = 'k')
    plt.quiver(*origin, *eig_vect1, color=['g'], scale=10)
    plt.quiver(*origin, *eig_vect2, color=['r'], scale=10)
    plt.title('Eigenvectors of Caovariance matrix')
    plt.savefig(base_dir + '/problem3_Cmat.png')
    plt.show()
    
    B = linear_ls(age, cost)
    C = ransac(age,cost)
    D = total_ls(cov_mat)
    x = np.linspace(np.min(age), np.max(age),500)
    y1 = B[0]*x + B[1]
    y2 = C[0]*x +C[1]
    d = D[0]*mean_age + D[1]*mean_cost
    y3 = (-D[0]*x + d)/D[1]
    plt.scatter(age,cost, c = 'k')
    plt.plot(x,y1,c = 'r', label = "Linear least square")
    plt.legend(loc="upper left")
    plt.title('Linear Least Square')
    plt.savefig(base_dir + '/problem3_lls.png')
    plt.show()
    
    
    plt.scatter(age,cost, c = 'k')
    plt.plot(x,y2, c = 'r',label = "RANSAC")
    plt.legend(loc="upper left")
    plt.title('RANSAC')
    plt.savefig(base_dir + '/problem3_ransac.png')
    plt.show()
    
    
    plt.scatter(age,cost, c = 'k')
    plt.plot(x,y3, c = 'r',label = "Total least square")
    plt.legend(loc="upper left")
    plt.title('Total Least Square')
    plt.savefig(base_dir + '/problem3_tls.png')
    plt.show()
    
    
if __name__ =='__main__':
    main()
