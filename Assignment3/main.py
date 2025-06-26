import numpy as np

from maraboupy import Marabou

def main():
    net = Marabou.read_onnx("iris_mlp.onnx")

    features = {
        0: 'sepal_length',
        1: 'sepal_width',
        2: 'petal_length',
        3: 'petal_width'
    }

    # Sample from Iris-setosa class: [5.1, 3.5, 1.4, 0.2]
    sample = [5.1, 3.5, 1.4, 0.2]
    epsilon = 0.05  # allowable perturbation
    
    input_vars = net.inputVars[0][0]
    
    # Set input bounds for each feature
    for i, val in enumerate(sample):
        lb = val - epsilon
        ub = val + epsilon
        net.setLowerBound(input_vars[i], lb)
        net.setUpperBound(input_vars[i], ub)

    # We expect the network to classify this perturbed input as class 0 (Iris-setosa)
    outVars = net.outputVars[0][0] # output logit variables for the 3 classes
    target_class = 0

    # Add inequalities: out[target] >= out[j] for all other classes j
    for j in range(len(outVars)):
        if j != target_class:
            net.addInequality([outVars[target_class], outVars[j]], [1.0, -1.0], 0.0)

    # Solve the verification query
    vals = net.solve()

    # Report results
    if vals[0] == 'sat':
        assignments = vals[1]  # 변수 ID -> 값
        print("Counterexample found within the epsilon-ball:")
        for i in range(4):
            var_idx = input_vars[i]
            print(f"{features[i]} = {assignments[var_idx]:.4f}")
        
        print("Output logits:")
        for i in range(len(outVars)):
            print(f"  out{i} = {assignments[outVars[i]]:.4f}")
    else:
        print("Property holds: no counterexample within the specified epsilon region.")



if __name__ == "__main__":
    main()
