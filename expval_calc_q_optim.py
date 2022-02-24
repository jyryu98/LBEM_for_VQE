import qiskit
import numpy as np
from qiskit import QuantumCircuit

def get_measuring_circuit(basis_list: list) -> QuantumCircuit:
    qc = QuantumCircuit(len(basis_list[0][1]))

    # Find a pauli word that has the most single qubit rotations
    basis = ''
    for qi in range(len(basis_list[0][1])):
        nonI = False
        for term in basis_list:
            if term[1][qi] != 'I':
                nonI = True
                basis += term[1][qi]
                break
        if not nonI:
            basis += 'I'
    
    # Construct the appropriate circuit
    for i,term in enumerate(basis):
        if term == 'X':
            qc.h(i)
        elif term == 'Y':
            qc.rx(np.pi/2,i)
        else:
            continue
    return qc

def pauli_expval(pauliwords: list, circuit: QuantumCircuit, qinstance: qiskit.utils.QuantumInstance, included_measuring_circuit = True) -> list:

    # If the basis rotation is not included, add appropriate rotations. All pauliwords should be commuting.
    if not included_measuring_circuit:
        templist = []
        for pauliword in pauliwords:
            templist.append((1, pauliword))
        measuring_circuit = get_measuring_circuit(templist)
        circuit = circuit.compose(measuring_circuit)
        if not qinstance.is_statevector:
            circuit.measure_all()

    # Run the ciruit!
    res = qinstance.execute(circuit)
    if qinstance.is_statevector:
        sv = res.get_statevector()
        pseudoprobs = sv.probabilities_dict()
    else:
        pseudoprobs = res.get_counts()

    # Calculate the expectation value of each observable
    expvals = []
    for pauliword in pauliwords:
        if not qinstance.is_statevector:
            total_counts = 0
        expval = 0
        for basis in pseudoprobs:
            if not qinstance.is_statevector:
                total_counts += pseudoprobs[basis]
            eigenvalue = 1
            for qubit, pauli in zip(basis[::-1], pauliword):
                if pauli != 'I' and qubit == '0':
                    eigenvalue = eigenvalue * 1
                if pauli != 'I' and qubit == '1':
                    eigenvalue = eigenvalue * (-1)
            expval += eigenvalue * pseudoprobs[basis]
        if not qinstance.is_statevector:
            expval = expval / total_counts
        expvals.append(expval)
    return expvals


def expval_calc(hamiltonian: list, circuits_to_run, em_instance: qiskit.utils.QuantumInstance, ef_instance: qiskit.utils.QuantumInstance):

    # Dictionaries to store com_ef and com_em results
    com_ef = {}
    com_em = {}
    
    # Calculate all com values
    for commuting_operators in hamiltonian:

        # Get measurement part of the circuit. This is the same for all commuting operators
        measurement_circuit = get_measuring_circuit(commuting_operators)

        for efcnum, efc_emcs in enumerate(circuits_to_run):
            efc = efc_emcs['efc']
            emcs = efc_emcs['emc']

            # Calculate all error free values
            # For each training circuit, add the measurement circuit and execute to find the statevector
            circuit_to_run = efc.compose(measurement_circuit)

            # Calculate the expectation value of each commuting operator
            pauliwords = []
            for coeff, pauliword in commuting_operators:
                pauliwords.append(pauliword)
            expvals = pauli_expval(pauliwords, circuit_to_run, ef_instance)

            # Save the result in com_ef
            for pauliword, expval in zip(pauliwords, expvals):
                com_ef[(pauliword, efcnum)] = expval

            # Calculate all noisy values
            # For each error mitigation circuit, add the measurement circuit and execute to find the shots
            for p in emcs:
                emc = emcs[p]
                circuit_to_run = emc.compose(measurement_circuit)
                circuit_to_run.measure_all()

                # Calculate the expectation value of each commuting operator
                expvals = pauli_expval(pauliwords, circuit_to_run, em_instance)

                # Save the result in com_em
                for pauliword, expval in zip(pauliwords, expvals):
                    com_em[(pauliword, efcnum, p)] = expval
    return com_ef, com_em

def q_optimize(hamiltonian: list, circuits_to_run, com_em: dict, com_ef: dict):

    # Calculate 'a' matrix and 'b' vector
    # Extend the definitions to include a constant q_0 term. a is 5 x 5 because of 4 paulis + q0 term
    extendedP = []
    for p in circuits_to_run[0]['emc']:
        extendedP.append(p)
    extendedP.append('q0')

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
    return q, extendedP