# Symmetric Clifford Twirling Simulation

Implementation of the symmetric Clifford twirling technique for quantum error mitigation as described in:

**"Symmetric Clifford twirling for cost-optimal quantum error mitigation in early FTQC regime"**  
*npj Quantum Information (2025) 11:104*  
Authors: Kento Tsubouchi, Yosuke Mitsuhashi, Kunal Sharma, Nobuyuki Yoshioka

## Overview

This implementation simulates the key results from the paper, including:

1. **Symmetric Clifford Twirling**: Converts local Pauli noise into noise resembling global white noise
2. **k-sparse Symmetric Clifford Twirling**: Hardware-efficient variant using only local operations
3. **Numerical Analysis**: Demonstrates performance on Trotterized Hamiltonian simulation circuits
4. **Error Mitigation**: Shows how to achieve cost-optimal quantum error mitigation with minimal sampling overhead

## Key Concepts

### What is Symmetric Clifford Twirling?

- **Problem**: Quantum gates are affected by noise, which degrades computation results
- **Challenge**: Cannot insert arbitrary operations around non-Clifford gates (like T gates or rotation gates)
- **Solution**: Use only Clifford operations that *commute* with the non-Clifford gate
- **Result**: Scrambles certain Pauli errors (X and Y) into global white noise, which can be mitigated optimally

### Main Results from the Paper

1. **Theorem 1**: Symmetric Clifford twirling can scramble Pauli-X and Y noise exponentially close to global white noise (distance ~ 2^(-n))
2. **Pauli-Z noise**: Remains unaffected by twirling (commutes with Z rotations)
3. **k-sparse twirling**: Achieves polynomial suppression (distance ~ n^(-(k-1)/2)) with only local operations
4. **Sampling overhead**: Cost-optimal error mitigation with overhead exp(2*p_tot) vs. exp(4*p_tot) for previous methods

## Files

### 1. `symmetric_clifford_twirling.py`
Main simulation script implementing:
- Symmetric Clifford operator generation
- k-sparse symmetric Clifford operators
- Trotterized Hamiltonian simulation circuits
- Bias analysis vs. qubit count
- Performance plots

### 2. `advanced_twirling_analysis.py`
Advanced analysis including:
- Exact Pauli noise channel representation
- Pauli propagation through Clifford gates
- Distance to white noise calculation
- Detailed comparison of twirling methods
- Scaling analysis with qubit count

## Installation

```bash
pip install qiskit numpy matplotlib scipy
```

## Usage

### Basic Simulation

```bash
python symmetric_clifford_twirling.py
```

This will:
1. Test symmetric Clifford twirling on various noise models
2. Demonstrate noise spreading effects
3. Generate analysis plots showing bias vs. qubit count

### Advanced Analysis

```bash
python advanced_twirling_analysis.py
```

This will:
1. Compare twirling methods on different noise models
2. Analyze scaling with number of qubits
3. Visualize noise spreading patterns
4. Calculate exact distances to white noise

## Key Results Reproduced

### Figure 4: Bias vs. Qubit Count
The simulation shows:
- **Without twirling**: Bias scales as 1/√n (white noise approximation)
- **With full twirling (X+Y noise)**: Exponential suppression ~ 2^(-n)
- **With 2-sparse twirling (X+Y noise)**: Polynomial suppression ~ 1/n
- **Depolarizing noise**: Similar performance for full and sparse (Z component dominates)

### Table 1: Distance to White Noise
| Noise Model | Initial | Full Twirling | 2-sparse Twirling |
|------------|---------|---------------|-------------------|
| X+Y only   | O(1)    | O(2^-n)       | O(n^-0.5)         |
| With Z     | O(1)    | p_z/p_err     | p_z/p_err         |

## Understanding the Code

### Symmetric Clifford Generation

```python
# Generate a symmetric Clifford operator for Z rotation on qubit 0
twirler = SymmetricCliffordTwirling(n_qubits=4)
symmetric_cliff = twirler.generate_symmetric_clifford(target_qubit=0)
```

The construction follows the paper's algorithm:
1. Probabilistically select target qubits (3/4 probability each)
2. Apply CNOT gates to propagate noise
3. Apply random single-qubit Cliffords to scramble
4. Apply S gate to control qubit (1/2 probability)

### k-sparse Twirling

```python
# Generate 2-sparse symmetric Clifford (affects at most 2 qubits)
sparse_cliff = twirler.generate_k_sparse_symmetric_clifford(k=2, target_qubit=0)
```

Benefits:
- Lower gate count
- Reduced additional errors
- Still achieves polynomial suppression for X+Y noise

### Noise Channel Analysis

```python
# Create Pauli noise channel
noise = create_single_qubit_pauli_noise(
    n_qubits=4, qubit=0, px=0.05, py=0.05, pz=0.0
)

# Apply twirling
twirled_noise = apply_symmetric_clifford_twirling(
    noise, n_samples=1000, target_qubit=0
)

# Calculate distance to white noise
distance = twirled_noise.distance_to_white_noise()
```

## Theoretical Background

### Cost-Optimal Error Mitigation

For global white noise with error rate p_err and L layers:
- Total error: p_tot = p_err × L
- Mitigation: Simply rescale expectation value by exp(p_tot)
- Sampling overhead: exp(2×p_tot) - **optimal**!

Compare to probabilistic error cancellation: exp(4×p_tot) - quadratically worse

### Why This Matters for Early FTQC

In the early fault-tolerant quantum computing regime:
- Logical qubits have residual noise
- Non-Clifford gates (T gates, rotations) dominate errors
- Error rates are ~10^-3 to 10^-6
- Circuit depth is limited

Symmetric Clifford twirling enables:
- Minimal sampling overhead for error mitigation
- Smaller code distances needed
- More efficient use of limited quantum resources

## Limitations and Extensions

### Current Limitations
1. Z-noise cannot be twirled (commutes with Z rotations)
2. Requires specific noise models (Pauli noise)
3. Clifford simulation used for large-scale demonstrations

### Future Directions (from paper)
1. Develop methods to perform non-Clifford gates with X/Y dominant noise
2. Apply to other contexts (fidelity estimation, black hole physics)
3. Combine with code switching to codes with transversal T gates

## Implementation Notes

### Clifford Simulation Mode
For large qubit counts, the code uses π/4 rotations (equivalent to S gates) to enable Clifford circuit simulation via stabilizer formalism. This allows:
- Efficient simulation of large systems
- Demonstration of noise scrambling effects
- Validation of theoretical predictions

### Pauli Propagation
The code tracks how Pauli operators propagate through Clifford gates using conjugation:
```
C P C† = P'
```
This is exact for Clifford gates and enables precise analysis.

## References

Main paper:
```
Tsubouchi, K., Mitsuhashi, Y., Sharma, K., & Yoshioka, N. (2025). 
Symmetric Clifford twirling for cost-optimal quantum error mitigation 
in early FTQC regime. npj Quantum Information, 11(104).
```

Related work:
- Probabilistic error cancellation: Temme et al., PRL 119, 180509 (2017)
- Optimal error mitigation bounds: Tsubouchi et al., PRL 131, 210601 (2023)
- Symmetric Clifford groups: Mitsuhashi & Yoshioka, PRX Quantum 4, 040331 (2023)