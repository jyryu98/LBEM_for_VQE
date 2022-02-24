def get_ansatz(n,m,ansatz = 'simple'):
    ## trivial circuit for H2 check might not work properly
    phi = Parameter('phi')
    qc = QuantumCircuit(n)
    if ansatz == 'simple':
        qc.x(0)
        qc.cx(0,1)
        qc.rx(phi,0)
        qc.cx(1,0)
        num_par_gate = 1
    elif ansatz == 'num_particle_preserving':
        qc, num_par_gate = n_qubit_A_circuit(n,m)
    else:
        raise NotImplementedError
    return qc, num_par_gate

def A_gate(qc, qubit1 , qubit2, theta):
    qc.cx(qubit2,qubit1)
    qc.ry(theta + math.pi/2,qubit2)
    qc.rz(math.pi,qubit2)
    qc.cx(qubit1,qubit2)
    qc.ry(theta + math.pi/2,qubit2)
    qc.rz(math.pi,qubit2)
    qc.cx(qubit2,qubit1)

def n_qubit_A_circuit(n,m, repeat = 1):
    qc = QuantumCircuit(n)
    index = 0
    theta  = ParameterVector('theta', repeat*(n-m)*m)
    ## primitive pattern
    for _ in range(repeat): 
        for i in range(m):
            qc.x(i)
            for j in range(m,n):
                A_gate(qc,i,j,theta[index])
                index += 1
    
    return qc, index
        




def main():
    
    molecule_name = 'LiH'
    distance = 1.4
    qubit_ham = calculate_pauli_hamiltonian(molecule_name, distance)['qubit_operator']
    optimized = optimize_measurements( [list(term[1]) for term in qubit_ham] )
    group_pauli_op = grouping(optimized,qubit_ham)
    ansatz, num_par_gates = get_ansatz(6,2,'num_particle_preserving')

    print(len(group_pauli_op))


    return group_pauli_op, (ansatz,num_par_gates)
