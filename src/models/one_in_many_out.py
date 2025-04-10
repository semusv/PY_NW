
def ele_mul(number,vector):
    output = [0,0,0]
    assert(len(output)==len(vector))

    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

def neural_network(input,weights):
    pred = ele_mul(input,weights)
    return pred

