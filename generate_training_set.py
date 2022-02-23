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
    
def get_circuits(qc):
    circuits_list = []

    for _ in range(len(clifford_list)):
        ef_em_set = []
        for _ in range(5):
            ef_em_set.append(QuantumCircuit(qc.width()))
        circuits_list.append(ef_em_set)
        
    for args in qc.data:
        print(args)
        if args[0].is_parameterized():
            print("here")
            qubit_idx = args[1][0].index
            for idx_c, gate_c in enumerate(clifford_list):
                for idx_p, gate_p in enumerate(pauli_list):
                    circuits_list[idx_c][idx_p+1].pauli(gate_p, [qubit_idx])
                for idx in range(5):
                    circuits_list[idx_c][idx] = circuits_list[idx_c][idx].compose(create_clifford(gate_c), [qubit_idx])
                
        else:
            print("here1")
            for idx_c, gate_c in enumerate(clifford_list):
                for i in range(5):
                    circuits_list[idx_c][i].data.append(args)

    return circuits_list
