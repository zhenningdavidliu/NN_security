import numpy as np

def shades2_alg(image):

    # Solves Task 1

    shades = np.sort(image.flatten())
    
    maxgap = 0
    secondgap = 0

    maxgaplower = 0
    secondgap_lower = 0

    for i in range(len(shades)-1):
        
        if maxgap < (shades[i+1] - shades[i]):

            secondgap = maxgap
            maxgap = shades[i+1]-shades[i]

            secondgap_lower = maxgaplower
            maxgaplower = i

        elif secondgap < (shades[i+1] - shades[i]):

            secondgap = shades[i+1] - shades[i]
            secondgap_lower = i


    if min(secondgap_lower, maxgaplower) == 63:

        label = 0

    else:

        label = 1

    return label



