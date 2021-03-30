# NNLO
Two neural network architectures for solving Hamilton-Jacobi PDEs without training. These architectures are based on the Lax-Oleinik formula.

The code corresponds to the paper https://doi.org/10.1016/j.jcp.2020.109907.
If you use our code in your research, please consider citing our paper:

J. Darbon and T. Meng, On some neural network architectures that can represent viscosity solutions of certain high dimensional Hamilton-Jacobi partial differential equations, Journal of Computational Physics, 425 (2021), p. 109907.

## How to run
### The first architecture (see section 3.1 in the paper)
LOformula1_1d.py gives an implementation of Example 3.1 in the paper, while LOformula1_hd.py gives an implementation of Example 3.2 in the paper. 
To run the codes for your own HJ PDE, do the following steps
- First, make sure your own HJ PDE satisfies the assumptions in Theorem 3.1 in the paper
- For one-dimensional problems (i.e., the spatial dimensiona is one), modify LOformula1_1d.py. For high-dimensional problems, modify LOformula1_hd.py.
- In the corresponding code, change L_fn to be the Legendre transform of your Hamiltonian, and change u_true and a_true to be the corresponding parameters (see the paper for the meaning of these parameters).

### The second architecture (see section 3.2 in the paper)
LOformula2_1d.py gives an implementation of Example 3.3 in the paper, while LOformula2_hd.py gives an implementation of Example 3.4 and 3.5 in the paper (by choosing different eg_no in the code). To run the codes for your own HJ PDE, do the following steps
- First, make sure your own HJ PDE satisfies the assumptions in Theorem 3.2 in the paper
- For one-dimensional problems (i.e., the spatial dimensiona is one), modify LOformula2_1d.py. For high-dimensional problems, modify LOformula2_hd.py.
- In the corresponding code, change J_fn to be the your initial condition J, and change v_true and b_true to be the corresponding parameters (see the paper for the meaning of these parameters).
