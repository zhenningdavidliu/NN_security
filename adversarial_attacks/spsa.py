# uses SPSA attack

import numpy as np


def project_interval(x, epsilon, y):

    # project y onto epsilon interval with center x
    
    if y>x: 
        out = min(y, x+epsilon)
    else:
        out = max(y, x-epsilon)

    return(out)

def project_ball(x, epsilon, y):

    # project y onto epsilon ball (l-infinity norm) with center x
    
    out = np.zeros(x.shape)
    w = x.shape # itterate for 2 dimensional input. Black and white images

    for i in range(w[0]):
        for j in range(w[1]):
            out[i,j] = project_interval(x[i,j], epsilon, y[i,j])

    return out

def trim_T1(x):

    # make sure that the images have values between 0 and 1

    out = np.zeros(x.shape)

    w = x.shape

    for i in range(w[0]):
        for j in range(w[1]):
            if x[i,j] > 1:
                out[i,j] = 1
            elif x[i,j] <0:
                out[i,j] = 0
            else:
                out[i,j] = x[i,j]

    return out

def spsa(f, x, delta, alpha, n, epsilon, T):

    # f     : function to minimize (Neural net)
    # x     : initial image to attack
    # delta : perturbation size
    # alpha : step size > 0
    # n     : batch size
    # T     : number of itterations
    v = []

    for i in range(n):
    
        v.append(np.random.randint(2, size = x.shape)*2 - 1) # Make sure to have the dimension of v and v_inv match x and the expected input for f

    v = np.array(v)
    v_inv = v*(-1)
    x_t = x
    
    g = [] 
    out = []

    for t in range(T):
   
        x1 = x_t + delta*v[i]
        x2 = x_t - delta*v[i]

        g_sum = np.zeros(x.shape)                                           # make sure this has same dimension as x 
        x1= np.expand_dims(x1, axis=0)
        x2= np.expand_dims(x2, axis=0)

        for i in range(n):
            g_sum += ((f.predict(x1) - f.predict(x2) ) * v_inv[i] /(2*delta))
            
        x_t2 = x_t - alpha*(1/n)*g_sum

        x_t = project_ball(x, epsilon, x_t2) # need to project x_t2 onto x(epsilon)

        out.append(x_t)

    return out

def spsa_T1(f, x, delta, alpha, n, epsilon, T):

    # f     : function to minimize (Neural net)
    # x     : initial image to attack
    # delta : perturbation size
    # alpha : step size > 0
    # n     : batch size
    # T     : number of itterations
    v = []

    for i in range(n):
    
        v.append(np.random.randint(2, size = x.shape)*2 - 1) # Make sure to have the dimension of v and v_inv match x and the expected input for f

    v = np.array(v)
    v_inv = v*(-1)
    x_t = x
    
    g = [] 
    out = []

    for t in range(T):
   
        x1 = x_t + delta*v[i]
        x2 = x_t - delta*v[i]

        g_sum = np.zeros(x.shape)                                           # make sure this has same dimension as x 
        x1= np.expand_dims(x1, axis=0)
        x2= np.expand_dims(x2, axis=0)

        for i in range(n):
            g_sum += ((f.predict(x1) - f.predict(x2) ) * v_inv[i] /(2*delta))
            
        x_t2 = x_t - alpha*(1/n)*g_sum

        x_t = project_ball(x, epsilon, x_t2) # need to project x_t2 onto x(epsilon)

        x_t = trim_T1(x_t)

        out.append(x_t)

    return out



