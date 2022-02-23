import qiskit
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise

def get_measuring_circuit(basis_list: list) -> QuantumCircuit:
    qc = QuantumCircuit(len(basis_list[0][1]))

    basis = ''
    for qi in range(len(basis_list[0][1])):
        nonI = False
        for term in basis_list:
            if term[1][qi] != 'I':
                nonI = True
                basis += term[1][qi]
        if not nonI:
            basis += 'I'

    for i,term in enumerate(basis):
        if term == 'X':
            qc.h(i)
        elif term == 'Y':
            qc.rx(np.pi/2,i)
        else:
            continue
    return qc

def expval_calc(hamiltonian, circuits_to_run):
    # Dictionaries to store com_ef and com_em results
    com_ef = {}
    com_em = {}

    # Quantum instances for error-free (ef) and noisy (em) experiments
    # Can be changed depending on experiment details ----------------------------------------------------------------#
    # Build noise model: Error probabilities
    prob_1 = 0.001  # 1-qubit gate
    prob_2 = 0.01   # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    qasm_simulator = qiskit.Aer.get_backend('qasm_simulator')
    em_instance = qiskit.utils.QuantumInstance(backend = qasm_simulator, noise_model = noise_model, shots = 10000)
    aer_simulator = qiskit.Aer.get_backend('aer_simulator_statevector')
    ef_instance = qiskit.utils.QuantumInstance(backend = aer_simulator)
    # ---------------------------------------------------------------------------------------------------------------#
    
    # Calculate all com values
    for commuting_operators in hamiltonian:

        # Get measurement part of the circuit. This is the same for all commuting operators
        measurement_circuit = get_measuring_circuit(commuting_operators)

        for efcnum, efc_emcs in enumerate(circuits_to_run):
            efc = efc_emcs['EFA']
            emcs = efc_emcs['EMA']

            # Calculate all error free values
            # For each training circuit, add the measurement circuit and execute to find the statevector
            circuit_to_run = efc.compose(measurement_circuit)
            res = ef_instance.execute(circuit_to_run)
            sv = res.get_statevector()
            probs = sv.probabilities_dict()

            # Calculate the expectation value of each commuting operator
            for coeff, pauliword in commuting_operators:
                expval = 0
                for basis in probs:
                    eigenvalue = 1
                    for qubit, pauli in zip(basis[::-1], pauliword):
                        if pauli != 'I' and qubit == '0':
                            eigenvalue = eigenvalue * 1
                        if pauli != 'I' and qubit == '1':
                            eigenvalue = eigenvalue * (-1)
                    expval += eigenvalue * probs[basis]

                # Save the result in com_ef
                com_ef[(pauliword, efcnum)] = expval

            # Calculate all noisy values
            # For each error mitigation circuit, add the measurement circuit and execute to find the shots
            for p in emcs:
                circuit_to_run = emcs[p].compose(measurement_circuit)
                circuit_to_run.measure_all()
                res = em_instance.execute(circuit_to_run)
                counts = res.get_counts()

                # Calculate the expectation value of each commuting operator
                for coeff, pauliword in commuting_operators:
                    expval = 0
                    total_counts = 0
                    for basis in counts:
                        total_counts += counts[basis]
                        eigenvalue = 1
                        for qubit, pauli in zip(basis[::-1], pauliword):
                            if pauli != 'I' and qubit == '0':
                                eigenvalue = eigenvalue * 1
                            if pauli != 'I' and qubit == '1':
                                eigenvalue = eigenvalue * (-1)
                        expval += eigenvalue * counts[basis]
                    expval = expval / total_counts

                    # Save the result in com_em
                    com_em[(pauliword, efcnum, p)] = expval
    return com_ef, com_em

def q_optimize(hamiltonian, circuits_to_run, com_em, com_ef):
    # Calculate 'a' matrix and 'b' vector
    # Extend the definitions to include a constant q_0 term. a is 5 x 5 because of 4 paulis + q0 term
    extendedP = ['I', 'X', 'Y', 'Z', 'q0']
    pauliwords = []
    for commuting_operators in hamiltonian:
        for coeff, pauliword in commuting_operators:
            pauliwords.append(pauliword)

    N = len(pauliwords)
    T = len(circuits_to_run)

    for pauliword in pauliwords:
        for r in range(T):
            com_em[(pauliword, r, 'q0')] = 1

    a = np.zeros((len(extendedP), len(extendedP)))
    b = np.zeros((len(extendedP)))

    for p1i, p1 in enumerate(extendedP):
        for p2i, p2 in enumerate(extendedP):
            for n in pauliwords:
                for r in range(T):
                    a[p1i][p2i] += com_em[(n, r, p1)]*com_em[(n, r, p2)]
            a[p1i][p2i] = a[p1i][p2i]/(N * T)

    for pi, p in enumerate(extendedP):
        for r in range(T):
            for pauliword in pauliwords:
                b[pi] += com_em[(pauliword, r, p)] * com_ef[(pauliword, r)]
        b[pi] = b[pi] / (N * T)

    # Optimize and find q vector
    q = np.dot(np.linalg.inv(a), b)
    return q