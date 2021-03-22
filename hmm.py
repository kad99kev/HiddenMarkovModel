import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz as gv


class HiddenMarkovModel:
    def __init__(
        self,
        observable_states,
        hidden_states,
        transition_matrix,
        emission_matrix,
        title="HMM",
    ):
        self.observable_states = observable_states
        self.hidden_states = hidden_states
        self.transition_matrix = pd.DataFrame(
            data=transition_matrix, columns=hidden_states, index=hidden_states
        )
        self.emission_matrix = pd.DataFrame(
            data=emission_matrix, columns=observable_states, index=hidden_states
        )
        self.pi = self._calculate_stationary_distribution()
        self.title = title

    def print_model_info(self):
        print("*" * 50)
        print(f"Observable States: {self.observable_states}")
        print(f"Emission Matrix:\n{self.emission_matrix}")
        print(f"Hidden States: {self.hidden_states}")
        print(f"Transition Matrix:\n{self.transition_matrix}")
        print(f"Initial Probabilities: {self.pi}")

    def visualize_model(self, output_dir="outputs", notebook=False):

        try:
            os.mkdir(output_dir)
        except FileExistsError:
            raise FileExistsError(
                "Directory already exists! Please provide a different output directory!"
            )
        output_loc = output_dir + "/" + self.title

        G = nx.MultiDiGraph()
        G.add_nodes_from(self.hidden_states)

        # Get transition probabilities
        hidden_edges = self._get_markov_edges(self.transition_matrix)
        for (origin, destination), weight in hidden_edges.items():
            G.add_edge(origin, destination, weight=weight, label=weight, color="blue")

        # Get emission probabilities
        emission_edges = self._get_markov_edges(self.emission_matrix)
        for (origin, destination), weight in emission_edges.items():
            G.add_edge(origin, destination, weight=weight, label=weight, color="red")

        # Create graph and draw with edge labels
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        edge_labels = {(n1, n2): d["label"] for n1, n2, d in G.edges(data=True)}
        nx.drawing.nx_pydot.write_dot(G, output_loc + ".dot")

        s = gv.Source.from_file(output_loc + ".dot", format="png")
        if notebook:
            from IPython.display import display

            display(s)
            return
        s.view()

    def forward(self, input_seq):

        input_seq = np.array(input_seq)
        n_states = len(self.hidden_states)
        T = len(input_seq)

        # Convert DataFrame to np.array
        emission_matrix = self.emission_matrix.values
        transition_matrix = self.transition_matrix.values

        # Initialize alpha
        alpha = np.zeros((n_states, T))
        alpha[:, 0] = self.pi * emission_matrix[:, input_seq[0]]

        for t in range(1, T):
            for s in range(n_states):
                alpha[s, t] = emission_matrix[s, input_seq[t]] * np.sum(
                    alpha[:, t - 1] * transition_matrix[:, s]
                )
        probs = alpha[:, -1].sum()
        return alpha, probs

    def backward(self, input_seq):

        input_seq = np.array(input_seq)
        n_states = len(self.hidden_states)
        T = len(input_seq)

        # Convert DataFrame to np.array
        emission_matrix = self.emission_matrix.values
        transition_matrix = self.transition_matrix.values

        # Initialize beta starting from last
        beta = np.zeros((n_states, T))
        beta[:, T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            for s in range(n_states):
                beta[s, t] = np.sum(
                    emission_matrix[:, input_seq[t + 1]]
                    * beta[:, t + 1]
                    * transition_matrix[s, :]
                )
        probs = sum(self.pi * emission_matrix[:, input_seq[0]] * beta[:, 0])
        return beta, probs

    def viterbi(self, input_seq):

        input_seq = np.array(input_seq)
        n_states = len(self.hidden_states)
        T = len(input_seq)

        # Convert DataFrame to np.array
        emission_matrix = self.emission_matrix.values
        transition_matrix = self.transition_matrix.values

        # Initial blank path
        path = np.zeros(T, dtype=int)
        # Delta = Highest probability of any path that reaches state i
        delta = np.zeros((n_states, T))
        # Phi = Argmax by time step for each state
        phi = np.zeros((n_states, T))

        # Initialize delta
        delta[:, 0] = self.pi * emission_matrix[:, input_seq[0]]

        print("*" * 50)
        print("Starting Forward Walk")

        for t in range(1, T):
            for s in range(n_states):
                delta[s, t] = (
                    np.max(delta[:, t - 1] * transition_matrix[:, s])
                    * emission_matrix[s, input_seq[t]]
                )
                phi[s, t] = np.argmax(delta[:, t - 1] * transition_matrix[:, s])
                print(f"State={s} : Sequence={t} | phi[{s}, {t}]={phi[s, t]}")

        print("*" * 50)
        print("Start Backtrace")
        path[T - 1] = np.argmax(delta[:, T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = phi[path[t + 1], [t + 1]]
            print(f"Path[{t}]={path[t]}")

        return path, delta, phi

    def _calculate_stationary_distribution(self):
        eig_vals, eig_vects = np.linalg.eig(self.transition_matrix.T.values)
        _eig_vects = eig_vects[:, np.isclose(eig_vals, 1)]
        _eig_vects = _eig_vects[:, 0]
        stationary = _eig_vects / _eig_vects.sum()
        stationary = stationary.real
        return stationary

    def _get_markov_edges(self, matrix):
        edges = {}
        for col in matrix.columns:
            for row in matrix.index:
                edges[(row, col)] = matrix.loc[row, col]
        return edges


def print_forward_result(alpha, a_prob):
    print("*" * 50)
    print(f"Alpha:\n{alpha}\nProbability of sequence: {a_prob}")


def print_backward_result(beta, b_prob):
    print("*" * 50)
    print(f"Beta:\n{beta}\nProbability of sequence: {b_prob}")


def print_viterbi_result(input_seq, observable_states, hidden_states, path, delta, phi):
    print("*" * 50)
    print("Viterbi Result")
    print(f"Delta:\n{delta}")
    print(f"Phi:\n{phi}")

    state_path = [hidden_states[p] for p in path]
    inv_input_seq = [observable_states[i] for i in input_seq]

    print(
        f"Result:\n{pd.DataFrame().assign(Observation=inv_input_seq).assign(BestPath=state_path)}"
    )
