Module hmm
==========

Functions
---------

    
`print_backward_result(beta, b_prob)`
:   Prints the result of the Backward Algorithm.
    
    Args:
        beta (np.array): A matrix of the beta values.
        b_prob (numpy.float64): The computed probability from the beta values.

    
`print_forward_result(alpha, a_prob)`
:   Prints the result of the Forward Algorithm.
    
    Args:
        alpha (np.array): A matrix of the alpha values.
        a_prob (numpy.float64): The computed probability from the alpha values.

    
`print_viterbi_result(input_seq, observable_states, hidden_states, path, delta, phi)`
:   Prints the result of the Viterbi Algorithm.
    
    Args:
        input_seq (list): A list of the observed input sequence.
        observable_states (list): A list containing the name of each observable state.
        hidden_states (list): A list containing the name of each hidden state.
        path (np.array): The output path for given input sequence.
        delta (np.array): A matrix of the delta values.
        phi (numpy.array): A matrix of the phi values.

Classes
-------

`HiddenMarkovModel(observable_states, hidden_states, transition_matrix, emission_matrix, title='HMM')`
:   Initialization function for HiddenMarkovModel
    
    Attributes:
        observable_states (list): A list containing the name of each observable state.
        hidden_states (list): A list containing the name of each hidden state.
        transition_matrix (2-D list): A matrix containing the transition probabilities.
        emission_matrix (2-D list): A matrix containing the emission probabilities.
        title (str): Title for the HMM project. Output files will be named with this attribute.

### Methods

`backward(self, input_seq)`
:   Runs the Backward Algorithm.

    Args:
        input_seq (list): A list of the observed input sequence.

    Returns:
        beta (np.array): A matrix of the beta values.
        probs (numpy.float64): The computed probability of the input sequence.

`forward(self, input_seq)`
:   Runs the Forward Algorithm.

    Args:
        input_seq (list): A list of the observed input sequence.

    Returns:
        alpha (np.array): A matrix of the alpha values.
        probs (numpy.float64): The computed probability of the input sequence.

`print_model_info(self)`
:   Prints the model in a readable manner.

`visualize_model(self, output_dir='outputs', notebook=False)`
:   Creates a transition and emission graph of the model.

    Args:
        output_dir (str): A directory will be created with this name. If the directory already exists then an error will be raised.
        notebook (bool): Whether the model should be visualized for a notebook or a script. If False, then a png will be displayed. If True then the output will be displayed in the IPython cell.

`viterbi(self, input_seq)`
:   Runs the Viterbi Algorithm.

    Args:
        input_seq (list): A list of the observed input sequence.

    Returns:
        path (np.array): The output path for given input sequence.
        delta (np.array): A matrix of the delta values.
        phi (numpy.array): A matrix of the phi values.
