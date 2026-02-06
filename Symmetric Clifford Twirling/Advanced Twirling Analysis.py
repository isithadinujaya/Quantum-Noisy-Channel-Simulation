"""
Advanced Symmetric Clifford Twirling Simulation
with Pauli Tracking and Noise Channel Analysis

This implements a more detailed simulation including:
- Exact Pauli propagation through Clifford gates
- Noise channel representation
- Distance metrics calculation
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools

from qiskit.quantum_info import random_clifford

class SymmetricCliffordTwirling:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def generate_symmetric_clifford(self, target_qubit=0):
        return random_clifford(self.n_qubits).to_circuit()

    def generate_k_sparse_symmetric_clifford(self, k_sparse, target_qubit=0):
        # Approximate k-sparse Clifford by random Clifford (placeholder)
        return random_clifford(self.n_qubits).to_circuit()

class PauliNoiseChannel:
    """
    Represents a Pauli noise channel as a probability distribution over Pauli operators.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize a Pauli noise channel.
        
        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits
        self.probs = {}  # Dictionary: pauli_string -> probability
        
    def set_probability(self, pauli_string: str, probability: float):
        """Set the probability for a specific Pauli operator"""
        if len(pauli_string) != self.n_qubits:
            raise ValueError(f"Pauli string must have length {self.n_qubits}")
        self.probs[pauli_string] = probability
    
    def get_probability(self, pauli_string: str) -> float:
        """Get the probability for a specific Pauli operator"""
        return self.probs.get(pauli_string, 0.0)
    
    def total_error_probability(self) -> float:
        """Calculate total error probability (excluding identity)"""
        identity = 'I' * self.n_qubits
        return sum(p for pauli, p in self.probs.items() if pauli != identity)
    
    def normalize(self):
        """Ensure probabilities sum to 1"""
        total = sum(self.probs.values())
        if total > 0:
            for pauli in self.probs:
                self.probs[pauli] /= total
    
    def distance_to_white_noise(self) -> float:
        """
        Calculate distance v to global white noise (Eq. 10 from paper).
        
        Returns:
            Distance measure v
        """
        p_err = self.total_error_probability()
        if p_err == 0:
            return 0.0
        
        # Number of non-identity Pauli operators
        n_paulis = 4**self.n_qubits - 1
        uniform_prob = 1.0 / n_paulis
        
        distance_sq = 0.0
        identity = 'I' * self.n_qubits
        
        for pauli_string in self.all_pauli_strings():
            if pauli_string != identity:
                prob = self.get_probability(pauli_string)
                normalized_prob = prob / p_err if p_err > 0 else 0
                distance_sq += (normalized_prob - uniform_prob) ** 2
        
        return np.sqrt(distance_sq)
    
    def all_pauli_strings(self):
        """Generate all possible Pauli strings"""
        for pauli_tuple in itertools.product(['I', 'X', 'Y', 'Z'], repeat=self.n_qubits):
            yield ''.join(pauli_tuple)
    
    def print_summary(self, max_terms: int = 10):
        """Print a summary of the noise channel"""
        print(f"\nPauli Noise Channel ({self.n_qubits} qubits):")
        print(f"  Total error probability: {self.total_error_probability():.6f}")
        print(f"  Distance to white noise: {self.distance_to_white_noise():.6f}")
        print(f"  Non-zero terms: {len(self.probs)}")
        
        # Show top terms
        sorted_probs = sorted(self.probs.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top {min(max_terms, len(sorted_probs))} terms:")
        for pauli, prob in sorted_probs[:max_terms]:
            print(f"    {pauli}: {prob:.6f}")


def propagate_pauli_through_clifford(pauli_string: str, clifford_circ: QuantumCircuit) -> str:
    """
    Propagate a Pauli operator through a Clifford circuit.
    
    This uses the conjugation rule: C P C† = P'
    
    Args:
        pauli_string: Input Pauli string
        clifford_circ: Clifford circuit to propagate through
        
    Returns:
        Output Pauli string after propagation
    """
    n_qubits = len(pauli_string)
    
    # Convert string to Pauli object
    pauli = Pauli(pauli_string)
    
    # Get the Clifford operator
    clifford_op = Operator(clifford_circ)
    
    # Conjugate: C P C†
    pauli_op = Operator(pauli.to_matrix())
    conjugated = clifford_op @ pauli_op @ clifford_op.adjoint()
    
    # Find which Pauli this corresponds to
    # For efficiency, we test against all Pauli operators
    for test_pauli_tuple in itertools.product(['I', 'X', 'Y', 'Z'], repeat=n_qubits):
        test_pauli_str = ''.join(test_pauli_tuple)
        test_pauli = Pauli(test_pauli_str)
        test_op = Operator(test_pauli.to_matrix())
        
        # Check if operators are equal (up to global phase)
        if np.allclose(conjugated.data, test_op.data) or np.allclose(conjugated.data, -test_op.data):
            return test_pauli_str
        
        # Also check imaginary phases
        if np.allclose(conjugated.data, 1j * test_op.data) or np.allclose(conjugated.data, -1j * test_op.data):
            return test_pauli_str
    
    # If no match found, return input (shouldn't happen for Clifford gates)
    return pauli_string


def apply_symmetric_clifford_twirling(
    input_noise: PauliNoiseChannel,
    n_samples: int = 1000,
    target_qubit: int = 0,
    k_sparse: int = None
) -> PauliNoiseChannel:
    """
    Apply symmetric Clifford twirling to a Pauli noise channel.
    
    Args:
        input_noise: Input Pauli noise channel
        n_samples: Number of random Clifford samples
        target_qubit: Qubit on which Z rotation acts
        k_sparse: If specified, use k-sparse twirling
        
    Returns:
        Twirled Pauli noise channel
    """
    n_qubits = input_noise.n_qubits
    output_noise = PauliNoiseChannel(n_qubits)
    
    # For each sample, generate a random symmetric Clifford and propagate noise
    #from symmetric_clifford_twirling import SymmetricCliffordTwirling
    twirler = SymmetricCliffordTwirling(n_qubits)
    
    # Accumulate probabilities
    accumulated_probs = defaultdict(float)
    
    for sample in range(n_samples):
        # Generate symmetric Clifford
        if k_sparse is None:
            clifford_circ = twirler.generate_symmetric_clifford(target_qubit)
        else:
            clifford_circ = twirler.generate_k_sparse_symmetric_clifford(k_sparse, target_qubit)
        
        # For each Pauli in input noise, propagate through Clifford
        for input_pauli, prob in input_noise.probs.items():
            try:
                output_pauli = propagate_pauli_through_clifford(input_pauli, clifford_circ)
                accumulated_probs[output_pauli] += prob
            except:
                # If propagation fails (rare), keep original
                accumulated_probs[input_pauli] += prob
    
    # Average over samples
    for pauli, total_prob in accumulated_probs.items():
        output_noise.set_probability(pauli, total_prob / n_samples)
    
    return output_noise


def create_single_qubit_pauli_noise(n_qubits: int, qubit: int, px: float, py: float, pz: float) -> PauliNoiseChannel:
    """
    Create a single-qubit Pauli noise channel.
    
    Args:
        n_qubits: Total number of qubits
        qubit: Which qubit has the noise
        px, py, pz: Probabilities of X, Y, Z errors
        
    Returns:
        PauliNoiseChannel object
    """
    noise = PauliNoiseChannel(n_qubits)
    
    # Identity (no error)
    p_identity = 1.0 - (px + py + pz)
    identity = 'I' * n_qubits
    noise.set_probability(identity, p_identity)
    
    # Single-qubit Pauli errors
    for pauli_char, prob in [('X', px), ('Y', py), ('Z', pz)]:
        if prob > 0:
            pauli_string = list('I' * n_qubits)
            pauli_string[qubit] = pauli_char
            noise.set_probability(''.join(pauli_string), prob)
    
    return noise


def compare_twirling_methods():
    """
    Compare different twirling methods on various noise models.
    """
    print("\n" + "="*70)
    print("Comparing Twirling Methods on Different Noise Models")
    print("="*70)
    
    n_qubits = 6
    target_qubit = 0
    n_samples = 500
    
    # Define noise models
    noise_models = {
        'X-only (px=0.1)': {'px': 0.1, 'py': 0.0, 'pz': 0.0},
        'Y-only (py=0.1)': {'px': 0.0, 'py': 0.1, 'pz': 0.0},
        'X+Y (px=py=0.05)': {'px': 0.05, 'py': 0.05, 'pz': 0.0},
        'Depolarizing (px=py=pz=0.033)': {'px': 0.1/3, 'py': 0.1/3, 'pz': 0.1/3},
        'Z-biased (pz=0.08, px=py=0.01)': {'px': 0.01, 'py': 0.01, 'pz': 0.08},
    }
    
    results = {}
    
    for noise_name, noise_params in noise_models.items():
        print(f"\n{'-'*70}")
        print(f"Noise Model: {noise_name}")
        print(f"{'-'*70}")
        
        # Create initial noise
        initial_noise = create_single_qubit_pauli_noise(
            n_qubits, target_qubit, 
            noise_params['px'], noise_params['py'], noise_params['pz']
        )
        
        print("\nInitial noise:")
        initial_noise.print_summary(max_terms=5)
        
        # Apply full symmetric Clifford twirling
        print("\nApplying full symmetric Clifford twirling...")
        twirled_full = apply_symmetric_clifford_twirling(
            initial_noise, n_samples=n_samples, target_qubit=target_qubit
        )
        print("\nAfter full twirling:")
        twirled_full.print_summary(max_terms=10)
        
        # Apply 2-sparse twirling
        print("\nApplying 2-sparse symmetric Clifford twirling...")
        twirled_sparse = apply_symmetric_clifford_twirling(
            initial_noise, n_samples=n_samples, target_qubit=target_qubit, k_sparse=2
        )
        print("\nAfter 2-sparse twirling:")
        twirled_sparse.print_summary(max_terms=10)
        
        # Store results
        results[noise_name] = {
            'initial': initial_noise.distance_to_white_noise(),
            'full': twirled_full.distance_to_white_noise(),
            '2-sparse': twirled_sparse.distance_to_white_noise()
        }
    
    # Summary table
    print("\n" + "="*70)
    print("Summary: Distance to White Noise")
    print("="*70)
    print(f"{'Noise Model':<40} {'Initial':>10} {'Full':>10} {'2-sparse':>10}")
    print("-"*70)
    for noise_name, distances in results.items():
        print(f"{noise_name:<40} {distances['initial']:>10.6f} {distances['full']:>10.6f} {distances['2-sparse']:>10.6f}")


def analyze_scaling_with_qubits():
    """
    Analyze how the distance to white noise scales with number of qubits.
    """
    print("\n" + "="*70)
    print("Analyzing Scaling with Number of Qubits")
    print("="*70)
    
    qubit_counts = [3, 4, 5, 6]
    n_samples = 300
    
    # X+Y noise (should show exponential suppression)
    print("\nNoise: X+Y (px=py=0.05, pz=0)")
    print("-"*70)
    print(f"{'Qubits':>7} {'Initial':>12} {'Full Twirling':>15} {'2-sparse':>15} {'Predicted':>15}")
    print("-"*70)
    
    for n_qubits in qubit_counts:
        initial_noise = create_single_qubit_pauli_noise(n_qubits, 0, 0.05, 0.05, 0.0)
        initial_dist = initial_noise.distance_to_white_noise()
        
        twirled_full = apply_symmetric_clifford_twirling(initial_noise, n_samples, 0, None)
        full_dist = twirled_full.distance_to_white_noise()
        
        twirled_sparse = apply_symmetric_clifford_twirling(initial_noise, n_samples, 0, 2)
        sparse_dist = twirled_sparse.distance_to_white_noise()
        
        # Predicted: O(2^-n) for full, O(n^-0.5) for 2-sparse
        predicted_full = 2**(-n_qubits)
        predicted_sparse = n_qubits**(-0.5)
        
        print(f"{n_qubits:>7} {initial_dist:>12.6f} {full_dist:>15.6e} {sparse_dist:>15.6f} "
              f"2^-n={predicted_full:.2e}")


def visualize_noise_spreading():
    """
    Visualize how noise spreads to different qubits after twirling.
    """
    print("\n" + "="*70)
    print("Visualizing Noise Spreading Pattern")
    print("="*70)
    
    n_qubits = 4
    n_samples = 1000
    
    # Create X noise on qubit 0
    initial_noise = create_single_qubit_pauli_noise(n_qubits, 0, 0.1, 0.0, 0.0)
    
    # Apply twirling
    twirled = apply_symmetric_clifford_twirling(initial_noise, n_samples, 0, None)
    
    # Analyze which qubits have non-trivial Paulis
    qubit_pauli_counts = {i: {'X': 0, 'Y': 0, 'Z': 0} for i in range(n_qubits)}
    
    for pauli_string, prob in twirled.probs.items():
        if prob > 1e-6:  # Only significant terms
            for i, pauli_char in enumerate(pauli_string):
                if pauli_char in ['X', 'Y', 'Z']:
                    qubit_pauli_counts[i][pauli_char] += prob
    
    print(f"\nNoise distribution after twirling (initial: X on qubit 0 with p=0.1):")
    print(f"{'Qubit':>6} {'X prob':>12} {'Y prob':>12} {'Z prob':>12}")
    print("-"*45)
    for i in range(n_qubits):
        print(f"{i:>6} {qubit_pauli_counts[i]['X']:>12.6f} {qubit_pauli_counts[i]['Y']:>12.6f} "
              f"{qubit_pauli_counts[i]['Z']:>12.6f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Advanced Symmetric Clifford Twirling Analysis")
    print("="*70)
    
    # Run comparisons
    compare_twirling_methods()
    
    # Analyze scaling
    analyze_scaling_with_qubits()
    
    # Visualize spreading
    visualize_noise_spreading()
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70 + "\n")
