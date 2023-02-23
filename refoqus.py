from typing import List, Tuple

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers


from scipy.stats import multinomial

""" Note that this code is adapted from the code example for Rosalin at:
https://pennylane.ai/qml/demos/tutorial_rosalin.html

"""


class Refoqus:
    """ Class for Refoqus Optimizer.
        
        Given one hamiltonian H = sum_{j=1}^M c_j h_j, and a set of states {rho_i}_{i=1}^N, we minimize an application-dependent loss function L:
        L(theta) = sum_{i} p_{i} l(E_{i}(theta))

        E_{i}(theta) is a measurable expectation and l is a linear function in this class.
        Here, we set the optimizer to work with StronglyEntanglingLayers for the variational part and the states are circuits obtained with VQE.
        Such dataset construction is explained at: https://pennylane.ai/qml/datasets.html.

        :param dataset_of_circuits: List of input state preparation circuits, {rho_i}_{i=1}^N.
        :param hamiltonian_terms: List of hamiltonian terms h_j making the hamiltonian_terms for one datapoint. 
        :param coeffs: Coefficient c_j for each hamiltonian term h_j.
        :param param_shape: Shape of the parameter array when applying StronglyEntanglingLayers.
        :param lr: Learning rate.
        :param min_shots: Minimal number of shots to distribute when estimating the loss.
        :param mu: Running average constant.
    """

    def __init__(
        self,
        dataset_of_circuits: List[qml.data.dataset.Dataset],
        hamiltonian_terms: List[qml.operation.Operator],
        coeffs: List[float],
        param_shape: Tuple[int],
        lr: float = 1.0,
        min_shots: int = 2,
        mu: float = 0.99,
    ):

        self.hamiltonian_terms = hamiltonian_terms
        self.coeffs = coeffs
        self.dataset_of_circuits = dataset_of_circuits
        self.nbdata = len(self.dataset_of_circuits)
        self.nbqbits = len(self.dataset_of_circuits[0].hamiltonian.wires)

        self.device = qml.device("default.qubit", wires=self.nbqbits, shots=100)

        # hyperparameters
        self.min_shots = min_shots
        self.mu = mu  # running average constant
        self.lr = lr  # learning rate

        # keep track of the total number of shots used
        self.shots_used = 0
        # total number of iterations
        self.iteration_count = 0
        # Number of shots per parameter
        self.s = np.zeros(param_shape, dtype=np.float64) + min_shots

        # Running average of the parameter gradients
        self.chi = None
        # Running average of the variance of the parameter gradients
        self.xi = None

        # sampling ingredients
        self.one_over_nbdata = 1.0 / self.nbdata
        self.prob_shots = [
            np.abs(c) * self.one_over_nbdata for c in self.coeffs
        ] * self.nbdata
        self.hamiltonian_terms_coeff_prob_shots = [
            (self.coeffs[j], self.hamiltonian_terms[j])
            for j in range(len(self.hamiltonian_terms))
        ] * self.nbdata
        self.indices_preparation_circuit = [
            i for i in range(self.nbdata) for c in self.coeffs
        ]

        self.lipschitz = np.sum(self.prob_shots)
        if lr > 2 / self.lipschitz:
            raise ValueError("The learning rate must be less than ", 2 / self.lipschitz)

    def estimate_cost(self, params, shots) -> np.ndarray:
        """ Estimate the cost function as an expectation value wrt the hamiltonian_terms terms for the current params with the number of shots being distributed among terms.

        	:param params: Current parameter values of the variational part.
        	:param shots: Total number of shots to be distributed among hamiltonian_terms terms.

        	:return: Numpy array with single-shot estimates for each term.
        """

        # construct the multinomial distribution, and sample
        # from it to determine how many shots to apply per term
        si = multinomial(n=shots, p=self.prob_shots)
        shots_per_term = si.rvs()[0]

        results = []

        @qml.qnode(self.device, diff_method="parameter-shift")
        def qnode(
            weights: np.ndarray,
            hamiltonian_terms: qml.operation.Operator,
            data_circuit: qml.data.dataset.Dataset,
        ):
            for op in data_circuit.vqe_gates:
                qml.apply(op)

            StronglyEntanglingLayers(weights, wires=self.device.wires)
            return qml.sample(hamiltonian_terms)

        for index_input_circuit, o_and_c, p, s in zip(
            self.indices_preparation_circuit,
            self.hamiltonian_terms_coeff_prob_shots,
            self.prob_shots,
            shots_per_term,
        ):

            if s > 0:
                c, o = o_and_c
                res = qnode(
                    params,
                    o,
                    self.dataset_of_circuits[index_input_circuit],
                    shots=int(s),
                )

                if s == 1:
                    res = np.array([res])

                results.append(res * c * self.one_over_nbdata / p)
        return np.concatenate(results)

    def evaluate_grad_var(
        self, i: int, params: np.ndarray, shots: int
    ) -> Tuple[np.ndarray]:
        """Evaluate the gradient, as well as the variance in the gradient, for the ith parameter in params, using the parameter-shift rule.

        	:param i: Index of the parameter to apply the shift rule.
        	:param params: Current parameter values of the variational part.
        	:param shots: Total number of shots to be distributed among hamiltonian_terms terms.

        	:return: Gradient and variance for the ith parameter in params.
        """
        shift = np.zeros_like(params)
        shift[i] = np.pi / 2

        shift_forward = self.estimate_cost(params + shift, shots)
        shift_backward = self.estimate_cost(params - shift, shots)

        g = np.mean(shift_forward - shift_backward) / 2
        s = np.var((shift_forward - shift_backward) / 2, ddof=1)

        return g, s

    def step(self, params):
        """Perform a single step of descent with Refoqus.

            :param params: Current parameter values of the variational part.
        """

        # keep track of the number of shots run
        self.shots_used += int(2 * np.sum(self.s))

        # compute the gradient, as well as the variance in the gradient,
        # using the number of shots determined by the array s.
        grad = []
        S = []

        p_ind = list(np.ndindex(*params.shape))

        for l in p_ind:
            # loop through each parameter, performing
            # the parameter-shift rule
            g_, s_ = self.evaluate_grad_var(l, params, self.s[l])
            grad.append(g_)
            S.append(s_)

        grad = np.reshape(np.stack(grad), params.shape)
        S = np.reshape(np.stack(S), params.shape)

        # gradient descent update
        params = params - self.lr * grad

        if self.xi is None:
            self.chi = np.zeros_like(params, dtype=np.float64)
            self.xi = np.zeros_like(params, dtype=np.float64)

        # running average of the gradient variance
        self.xi = self.mu * self.xi + (1 - self.mu) * S
        xi = self.xi / (1 - self.mu ** (self.iteration_count + 1))

        # running average of the gradient
        self.chi = self.mu * self.chi + (1 - self.mu) * grad
        chi = self.chi / (1 - self.mu ** (self.iteration_count + 1))

        # determine the new optimum shots distribution for the next
        # iteration of the optimizer
        s = np.ceil(
            (2 * self.lipschitz * self.lr)
            * np.sqrt(S)
            * np.sum(np.sqrt(S))
            / (np.sum(grad ** 2))
            / (2 - self.lipschitz * self.lr)
        )
        self.s = np.clip(s, min(2, self.min_shots), None)

        self.iteration_count += 1
        return params
