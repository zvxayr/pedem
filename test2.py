from turtle import width
import numpy as np
import matplotlib.pyplot as plt

grad_direction = [[0,2.5,3],
        [2,0.4,1],
        [0.5,2,2]]

grad_magnitude = [[0,90,3],
        [0,80,0],
        [0,0,0]]

grad_magnitude = np.array(grad_magnitude)
grad_direction = np.array(grad_direction)
output = np.zeros(grad_magnitude.shape)

height, width = grad_magnitude.shape
pi = np.pi

for i_x in range(width):
    for i_y in range(height):
        try:
            if (grad_direction[i_y, i_x] < (pi / 8)) and (grad_direction[i_y, i_x] >= (-pi / 8)) or (grad_direction[i_y, i_x] < (9 * pi / 8)) and (grad_direction[i_y, i_x] >= (7 * pi / 8)) or (grad_direction[i_y, i_x] < (-9 * pi / 8)) and (grad_direction[i_y, i_x] >= (-7 * pi / 8)):
                if (grad_magnitude[i_y + 1, i_x] > grad_magnitude[i_y, i_x]) or (grad_magnitude[i_y - 1, i_x] > grad_magnitude[i_y, i_x]):
                    output[i_y,i_x] = 0
                else:
                    output[i_y,i_x] = grad_magnitude[i_y, i_x]

            elif (grad_direction[i_y, i_x] < (3 * pi / 8)) and (grad_direction[i_y, i_x] >= (pi / 8)) or (grad_direction[i_y, i_x] < (11 * pi / 8)) and (grad_direction[i_y, i_x] >= (9 * pi / 8)) or (grad_direction[i_y, i_x] < (-7 * pi / 8)) and (grad_direction[i_y, i_x] >= (-5 * pi / 8)) or (grad_direction[i_y, i_x] < (-15 * pi / 8)) and (grad_direction[i_y, i_x] >= (-13 * pi / 8)):
                if (grad_magnitude[i_y + 1, i_x + 1] > grad_magnitude[i_y, i_x]) or (grad_magnitude[i_y - 1, i_x - 1] > grad_magnitude[i_y, i_x]):
                    output[i_y,i_x] = 0
                else:
                    output[i_y,i_x] = grad_magnitude[i_y, i_x]
                    
            elif (grad_direction[i_y, i_x] < (5 * pi / 8)) and (grad_direction[i_y, i_x] >= (3 * pi / 8)) or (grad_direction[i_y, i_x] < (13 * pi / 8)) and (grad_direction[i_y, i_x] >= (11 * pi / 8)) or (grad_direction[i_y, i_x] < (-5 * pi / 8)) and (grad_direction[i_y, i_x] >= (-3 * pi / 8)) or (grad_direction[i_y, i_x] < (-13 * pi / 8)) and (grad_direction[i_y, i_x] >= (-11 * pi / 8)):
                if (grad_magnitude[i_y, i_x + 1] > grad_magnitude[i_y, i_x]) or (grad_magnitude[i_y, i_x - 1] > grad_magnitude[i_y, i_x]):
                    output[i_y,i_x] = 0
                else:
                    output[i_y,i_x] = grad_magnitude[i_y, i_x]

            else:
                if  (grad_magnitude[i_y - 1, i_x + 1] > grad_magnitude[i_y, i_x]) or (grad_magnitude[i_y + 1, i_x - 1] > grad_magnitude[i_y, i_x]):
                    output[i_y,i_x] = 0
                else:
                    output[i_y,i_x] = grad_magnitude[i_y, i_x]
        except:
            END
print(output)
