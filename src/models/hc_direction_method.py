weight = 0.5
input = 0.5
goal_prediction = 0.8
step_amount = 0.001

for iteration in range(50):
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2

    direction_and_amount=(prediction-goal_prediction)*input

    print("Error: "+str(error) + "   Prediction: " + str(prediction) + "  Weight: " + str(weight)  + " direction_and_amount: " + str(direction_and_amount) )
    weight = weight - direction_and_amount

