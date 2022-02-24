from qiskit import *

clifford_list = ['I','X','Y','Z','S','XS','YS','ZS','H','XH','YH','ZH','SH','XSH','YSH','ZSH','HS','XHS','YHS','ZHS','SHS','XSHS','YSHS','ZSHS']
pauli_list = ['I','X','Y','Z']

def create_clifford(str):
    clifford_circuit = QuantumCircuit(1)
    for c in str:
        if c == 'I':
            clifford_circuit.i(0)
        elif c == 'X':
            clifford_circuit.x(0)
        elif c == 'Y':
            clifford_circuit.y(0)
        elif c == 'Z':
            clifford_circuit.z(0)
        elif c == 'H':
            clifford_circuit.h(0)
        elif c == 'S':
            clifford_circuit.s(0)
    return clifford_circuit

def insert_pauli(qc, pauli):
    new_circuit = QuantumCircuit(qc.width())
 
    for args in qc.data:
        if args[0].is_parameterized():
            qubit_idx = args[1][0].index
            new_circuit.pauli(pauli[0], [qubit_idx])
            new_circuit.data.append(args)
            pauli = pauli[1:]
        else:
            new_circuit.data.append(args)
 
    return new_circuit

def get_pauli_comb(n):
    pauli_comb_list = pauli_list.copy()
 
    for _ in range(n-1):
        pauli_comb_list = [pauli1 + pauli2 for pauli1 in pauli_comb_list for pauli2 in pauli_list]

    return pauli_comb_list    

def get_circuits_dict(qc):
    circuits_list = []
    pauli_comb_list = get_pauli_comb(qc.num_parameters)

    for args in clifford_list:
        ef_em_dict = {}
        ef_em_dict['efc'] = QuantumCircuit(qc.width())
 
        em_dict = {}
        for pauli in pauli_comb_list:
            em_dict[pauli] = QuantumCircuit(qc.width())
        ef_em_dict['emc'] = em_dict
 
        circuits_list.append(ef_em_dict)
 
    pauli_idx = 0
    for args in qc.data:
        if args[0].is_parameterized():
            qubit_idx = args[1][0].index
            for idx_c, gate_c in enumerate(clifford_list):
                for pauli_comb in pauli_comb_list:
                    circuits_list[idx_c]['emc'][pauli_comb].pauli(pauli_comb[pauli_idx], [qubit_idx])
 
                clifford_to_add = create_clifford(gate_c)
                circuits_list[idx_c]['efc'] = circuits_list[idx_c]['efc'].compose(clifford_to_add, [qubit_idx])
                for pauli_comb in pauli_comb_list:
                    circuits_list[idx_c]['emc'][pauli_comb] = circuits_list[idx_c]['emc'][pauli_comb].compose(clifford_to_add, [qubit_idx])
 
            pauli_idx = pauli_idx + 1
 
        else:
            for idx_c, gate_c in enumerate(clifford_list):
                circuits_list[idx_c]['efc'].data.append(args)
                for pauli_comb in pauli_comb_list:
                    circuits_list[idx_c]['emc'][pauli_comb].data.append(args)

    return circuits_list

def get_circuit_list(qc):
    circuits_list = []

    for _ in range(len(clifford_list)):
        ef_em_set = []
        for _ in range(5):
            ef_em_set.append(QuantumCircuit(qc.width()))
        circuits_list.append(ef_em_set)
 
    for args in qc.data:
        if args[0].is_parameterized():
            qubit_idx = args[1][0].index
            for idx_c, gate_c in enumerate(clifford_list):
                for idx_p, gate_p in enumerate(pauli_list):
                    circuits_list[idx_c][idx_p+1].pauli(gate_p, [qubit_idx])
                for idx in range(5):
                    circuits_list[idx_c][idx] = circuits_list[idx_c][idx].compose(create_clifford(gate_c), [qubit_idx])
 
        else:
            for idx_c, gate_c in enumerate(clifford_list):
                for i in range(5):
                    circuits_list[idx_c][i].data.append(args)

    return circuits_list
