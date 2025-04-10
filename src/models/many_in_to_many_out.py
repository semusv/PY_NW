# import src.models.many_in_to_many_out as nw
    
# weights = [
    
# #    Игр  Побед  Болельщик
#     [0.1,  0.1,   -0.3],  # Веса для "травмы?"
#     [0.1,  0.2,    0.0],  # Веса для "победы?"
#     [0.0,  1.3,    0.1]   # Веса для "печали?"
# ]
# toes  = [8.5 , 9.5, 9.9, 9.0]
# wlrec = [0.65, 0.8, 0.8, 0.9]
# nfans = [1.2 , 1.3, 0.5, 1.0]

# for i in range(len(toes)):
#     input = [toes[i],wlrec[i],nfans[i]]
#     pred = nw.neural_network(input,weights)
#     print(pred)
import numpy as np



def neural_network(input, weights, sigm):
    pred = vect_mat_mul(input,weights, sigm)
    return pred

def vect_mat_mul(vect,matrix, sigm):
    assert(len(vect)==len(matrix))
    output=[0,0,0]

    for i in range(len(vect)):
        output[i]=w_sum(vect,matrix[i], sigm)
    return output

def w_sum(a,b, sigm):
    assert(len(a)==len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i]*b[i])
    if (sigm is True ):
        output = sigmoid(output)
    return output

def sigmoid(x):
    return round(float(1 / (1 + np.exp(-x))),4)


