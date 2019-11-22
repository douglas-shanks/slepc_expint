# slepc_expint
Slight modification of a slepc mfn example (will build on this later).

Compile: For example, assuming a prexisting build of PETSc and SLEPc

```python
>> cd slepc_expint
>> export PETSC_DIR=/home/dshanks/SLEPc_build/petsc
>> export PETSC_ARCH=arch-linux2-c-debug
>> SLEPC_DIR=/home/dshanks/SLEPc_build/slepc
>> make ex39_mod
```
Run: 

```python
>> OMP_NUM_THREADS=1  mpiexec -np 1 ./slepc_expint
>>
Heat equation via phi functions, n=99, tend=1., deltat=0.1

 The relative error at t=1. is 0.0571

 Number of iterations of the method: 82
 Stopping condition: tol=1e-08, maxit=100
 Number of time steps: 10
Elapsed time is 19.056227


```
