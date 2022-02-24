
import matplotlib.pyplot as plt
import numpy as np
from pyrsistent import m
import qiskit
import math

import warnings
warnings.filterwarnings("ignore")

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
#from qiskit_nature.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry import FermionicOperator
from qiskit.aqua.operators import Z2Symmetries

from qiskit.algorithms.optimizers import SLSQP,POWELL,SPSA,COBYLA
from qiskit.circuit.library import TwoLocal

from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit import IBMQ, BasicAer, Aer
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit, Parameter

from qiskit.circuit import ParameterVector, QuantumCircuit



def calculate_pauli_hamiltonian(molecule_name, distance ,map_type = 'parity'):
  
    if molecule_name =='H2':
        freeze_list = []
        remove_list = []
        molecular_coordinates = "H 0 0 0; H 0 0 " + str(distance)
    elif molecule_name == 'LiH':
        freeze_list = [0,6]
        remove_list = [4,8]
        molecular_coordinates = "Li 0 0 0; H 0 0 " + str(distance)
    else:
        raise NotImplementedError


    driver = PySCFDriver(molecular_coordinates, unit=UnitsType.ANGSTROM,charge=0,spin=0,basis='sto3g')
    molecule = driver.run()
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)

    ##  number of oribials and particles
    num_spin_orbitals = molecule.num_orbitals * 2
    num_particles = molecule.num_alpha + molecule.num_beta
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy

    ## freeze the core orbitals
    if(freeze_list == []):
        ferOp_f = ferOp
        energy_shift = 0
    else:
        ferOp_f, energy_shift = ferOp.fermion_mode_freezing(freeze_list)

    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)

    ## remove the determined orbitals
    if(remove_list == []):
        ferOp_fr = ferOp_f
    else:
        ferOp_fr = ferOp_f.fermion_mode_elimination(remove_list)
    
    num_spin_orbitals -= len(remove_list)

    ### get  the qubit hamiltonian
    qubitOp = ferOp_fr.mapping(map_type=map_type)

    ### reduce number of qubits through z2 symmetries
    if map_type == 'parity':
        qubitOp_t = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    else:
        qubitOp_t = qubitOp
    
    ## process the qubit operator ############
    qubit_ham = []
    for coeff,pauli in qubitOp_t.__dict__['_paulis']:
        if all([p == "I" for p in pauli.to_label()]):
            identity_coeff = coeff
            continue
        qubit_ham.append((coeff,pauli.to_label()))

    result = {'qubit_operator': qubit_ham,'coeff_identity_pauli': identity_coeff , 'shift':nuclear_repulsion_energy + energy_shift,'num_particles': num_particles,'num_spin_orbitals':num_spin_orbitals }

    return result


def check_simplification(op1, op2):
    """As we have seen in the problem statement, given two Pauli operators, you could obtain the expected value
    of each of them by running a single circuit following the two defined rules. This function will determine whether,
    given two Pauli operators, such a simplification can be produced.

    Args:
        - op1 (list(str)): First Pauli word (list of Pauli operators), e.g., ["Y", "I", "Z", "I"].
        - op2 (list(str)): Second Pauli word (list of Pauli operators), e.g., ["Y", "I", "X", "I"].

    Returns:
        - (bool): 'True' if we can simplify them, 'False' otherwise. For the example args above, the third qubit does not allow simplification, so the function would return `False`.
    """

    # QHACK
    for i in range(len(op1)):
        if ( op1[i]!= op2[i] ) and  "I"  not in [ op1[i], op2[i]] :
            return False
    return True
    # QHACK



def join_operators(op1, op2):
    """This function will receive two operators that we know can be simplified
    and returns the operator corresponding to the union of the two previous ones.

    Args:
        - op1 (list(str)): First Pauli word (list of Pauli operators), e.g., ["Y", "I", "Z", "I"].
        - op2 (list(str)): Second Pauli word (list of Pauli operators), e.g., ["Y", "X", "I", "I"].

    Returns:
        - (list(str)): Pauli operator corresponding to the union of op1 and op2.
        For the case above the output would be ["Y", "X", "Z", "I"]
    """

    # QHACK
    assert(check_simplification(op1,op2))
    joined_op = []
    for i in range(len(op1)):
        if op1[i] == op2[i]:
            joined_op.append(op1[i])
        else:
            joined_op.append(op1[i] if op1[i] != "I" else op2[i])

    return joined_op


    # QHACK


def optimize_measurements(obs_hamiltonian):
    """This function will go through the list of Pauli words provided in the statement, grouping the operators
    following the simplification process of the previous functions.

    Args:
        - obs_hamiltonian (list(list(str))): Groups of Pauli words making up the Hamiltonian.

    Returns:
        - (list(list(str))): The chosen Pauli operators to measure after grouping.
    """
    
    final_solution = []
    grouped_dic = {}
    for op1 in obs_hamiltonian:
        added = False
        for i, op2 in enumerate(final_solution):

            if check_simplification(op1, op2):
                final_solution[i] = join_operators(op1, op2)
                added = True
                break
        if not added:
            final_solution.append(op1)
        
    return final_solution

def grouping(final_solution, obs_hamiltonian ):
    
    print(final_solution)
    grouped_list = []
    for i,meas_op in enumerate(final_solution):
        temp = []
        for coeff,operator in obs_hamiltonian:
            if(check_simplification(list(operator),meas_op)):
                temp.append((coeff,operator))
        grouped_list.append(temp)
    return grouped_list


def get_ansatz(n,m,ansatz = 'simple'):
    ## trivial circuit for H2 check might not work properly
    phi = Parameter('phi')
    qc = QuantumCircuit(n)
    if ansatz == 'simple':
        qc.x(0)
        qc.cx(0,1)
        qc.rx(phi,0)
        qc.cx(1,0)
    elif ansatz == 'num_particle_preserving':
        qc = n_qubit_A_circuit(n,m)
    else:
        raise NotImplementedError
    return qc


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
    
    return qc
        


def main():
    
    molecule_name = 'LiH'
    distance = 1.4
    qubit_ham = calculate_pauli_hamiltonian(molecule_name, distance)['qubit_operator']
    optimized = optimize_measurements( [list(term[1]) for term in qubit_ham] )
    group_pauli_op = grouping(optimized,qubit_ham)
    ansatz = get_ansatz(6,2,'num_particle_preserving')



    return group_pauli_op, ansatz

