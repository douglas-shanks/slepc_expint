/*
   This example illustrates the use of Phi functions in exponential integrators.
   In particular, it implements the Norsett-Euler scheme of stiff order 1.

   The problem is the 1-D heat equation with source term

             y_t = y_xx + 1/(1+u^2) + psi

   where psi is chosen so that the exact solution is yex = x*(1-x)*exp(tend).
   The space domain is [0,1] and the time interval is [0,tend].

       [1] M. Hochbruck and A. Ostermann, "Explicit exponential Runge-Kutta
           methods for semilinear parabolic problems", SIAM J. Numer. Anal. 43(3),
           1069-1090, 2005.
*/

static char help[] = "Exponential integrator for the heat equation with source term.\n\n"
  "The command line options are:\n"
  "  -n <idim>, where <idim> = dimension of the spatial discretization.\n"
  "  -tend <rval>, where <rval> = real value that corresponding to the final time.\n"
  "  -deltat <rval>, where <rval> = real value for the time increment.\n"
  "  -combine <bool>, to represent the phi function with FNCOMBINE instead of FNPHI.\n\n";

#include <slepcmfn.h>

/*
   BuildFNPhi: builds an FNCOMBINE object representing the phi_1 function

        f(x) = (exp(x)-1)/x

   with the following tree:

            f(x)                  f(x)              (combined by division)
           /    \                 p(x) = x          (polynomial)
        a(x)    p(x)              a(x)              (combined by addition)
       /    \                     e(x) = exp(x)     (exponential)
     e(x)   c(x)                  c(x) = -1         (constant)
*/
PetscErrorCode BuildFNPhi(FN fphi)
{
  PetscErrorCode ierr;
  FN             fexp,faux,fconst,fpol;
  PetscScalar    coeffs[2];

  PetscFunctionBeginUser;
  ierr = FNCreate(PETSC_COMM_WORLD,&fexp);CHKERRQ(ierr);
  ierr = FNCreate(PETSC_COMM_WORLD,&fconst);CHKERRQ(ierr);
  ierr = FNCreate(PETSC_COMM_WORLD,&faux);CHKERRQ(ierr);
  ierr = FNCreate(PETSC_COMM_WORLD,&fpol);CHKERRQ(ierr);

  ierr = FNSetType(fexp,FNEXP);CHKERRQ(ierr);

  ierr = FNSetType(fconst,FNRATIONAL);CHKERRQ(ierr);
  coeffs[0] = -1.0;
  ierr = FNRationalSetNumerator(fconst,1,coeffs);CHKERRQ(ierr);

  ierr = FNSetType(faux,FNCOMBINE);CHKERRQ(ierr);
  ierr = FNCombineSetChildren(faux,FN_COMBINE_ADD,fexp,fconst);CHKERRQ(ierr);

  ierr = FNSetType(fpol,FNRATIONAL);CHKERRQ(ierr);
  coeffs[0] = 1.0; coeffs[1] = 0.0;
  ierr = FNRationalSetNumerator(fpol,2,coeffs);CHKERRQ(ierr);

  ierr = FNSetType(fphi,FNCOMBINE);CHKERRQ(ierr);
  ierr = FNCombineSetChildren(fphi,FN_COMBINE_DIVIDE,faux,fpol);CHKERRQ(ierr);

  ierr = FNDestroy(&faux);CHKERRQ(ierr);
  ierr = FNDestroy(&fpol);CHKERRQ(ierr);
  ierr = FNDestroy(&fconst);CHKERRQ(ierr);
  ierr = FNDestroy(&fexp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat                L;
  Vec                u,w,z,yex;
  MFN                mfnexp,mfnphi;
  FN                 fexp,fphi;
  PetscBool          combine=PETSC_FALSE;
  PetscInt           i,k,Istart,Iend,n=99,steps, its,totits=0,maxit;
  PetscReal          t,tend=1.0,deltat=0.1,nrmd,nrmu,x,h,tol;
  PetscScalar        value,c,uval,*warray;
  const PetscScalar *uarray;
  PetscErrorCode     ierr;
  PetscMPIInt    npe,mype;
  double t1, t2;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-tend",&tend,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-deltat",&deltat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-combine",&combine,NULL);CHKERRQ(ierr);
  h = 1.0/(n+1.0);
  c = (n+1)*(n+1);

  ierr  = MPI_Comm_rank(PETSC_COMM_WORLD, &mype);CHKERRQ(ierr);
  ierr  = MPI_Comm_size(PETSC_COMM_WORLD, &npe);CHKERRQ(ierr);
  
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nHeat equation via phi functions, n=%D, tend=%g, deltat=%g%s\n\n",n,(double)tend,(double)deltat,combine?" (combine)":"");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Build the 1-D Laplacian and various vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  
  ierr = MatCreate(PETSC_COMM_WORLD,&L);CHKERRQ(ierr);
  ierr = MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(L);CHKERRQ(ierr);
  ierr = MatSetUp(L);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(L,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;++i) {
    if (i>0) { ierr = MatSetValue(L,i,i-1,c,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n-1) { ierr = MatSetValue(L,i,i+1,c,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(L,i,i,-2.0*c,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateVecs(L,NULL,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&yex);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&w);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&z);CHKERRQ(ierr);

  /*
     Compute various vectors:
     - the exact solution yex = x*(1-x)*exp(tend)
     - the initial condition u = abs(x-0.5)-0.5
  */
  ierr = MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
  t1 = MPI_Wtime(); // start timer
  for (i=Istart;i<Iend;++i) {
    x = (i+1)*h;
    value = x*(1.0-x)*PetscExpReal(tend);
    ierr = VecSetValue(yex,i,value,INSERT_VALUES);CHKERRQ(ierr);
    value = PetscAbsReal(x-0.5)-0.5;
    ierr = VecSetValue(u,i,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(yex);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(yex);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(yex,NULL,"-exact_sol");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u,NULL,"-initial_cond");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create two MFN solvers, for exp() and phi_1()
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MFNCreate(PETSC_COMM_WORLD,&mfnexp);CHKERRQ(ierr);
  ierr = MFNSetOperator(mfnexp,L);CHKERRQ(ierr);
  ierr = MFNGetFN(mfnexp,&fexp);CHKERRQ(ierr);
  ierr = FNSetType(fexp,FNEXP);CHKERRQ(ierr);
  ierr = FNSetScale(fexp,deltat,1.0);CHKERRQ(ierr);
  ierr = MFNSetErrorIfNotConverged(mfnexp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MFNSetFromOptions(mfnexp);CHKERRQ(ierr);

  ierr = MFNCreate(PETSC_COMM_WORLD,&mfnphi);CHKERRQ(ierr);
  ierr = MFNSetOperator(mfnphi,L);CHKERRQ(ierr);
  ierr = MFNGetFN(mfnphi,&fphi);CHKERRQ(ierr);
  if (combine) {
    ierr = BuildFNPhi(fphi);CHKERRQ(ierr);
  } else {
    ierr = FNSetType(fphi,FNPHI);CHKERRQ(ierr);
    ierr = FNPhiSetIndex(fphi,1);CHKERRQ(ierr);
  }
  ierr = FNSetScale(fphi,deltat,1.0);CHKERRQ(ierr);
  ierr = MFNSetErrorIfNotConverged(mfnphi,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MFNSetFromOptions(mfnphi);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Solve the problem with the Norsett-Euler scheme
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  steps = PetscRoundReal(tend/deltat);
  t = 0.0;
  for (k=0;k<steps;++k) {

    /* evaluate nonlinear part */
    ierr = VecGetArrayRead(u,&uarray);CHKERRQ(ierr);
    ierr = VecGetArray(w,&warray);CHKERRQ(ierr);
    for (i=Istart;i<Iend;++i) {
      x = (i+1)*h;
      uval = uarray[i-Istart];
      value = x*(1.0-x)*PetscExpReal(t);
      value = value + 2.0*PetscExpReal(t) - 1.0/(1.0+value*value);
      value = value + 1.0/(1.0+uval*uval);
      warray[i-Istart] = deltat*value;
    }
    ierr = VecRestoreArrayRead(u,&uarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(w,&warray);CHKERRQ(ierr);
    ierr = MFNSolve(mfnphi,w,z);CHKERRQ(ierr);

    /* evaluate linear part */
    ierr = MFNSolve(mfnexp,u,u);CHKERRQ(ierr);
    ierr = MFNGetIterationNumber(mfnexp,&its);CHKERRQ(ierr);
    ierr = MFNGetTolerances(mfnexp,&tol,&maxit);CHKERRQ(ierr);
    totits += its;
    ierr = VecAXPY(u,1.0,z);CHKERRQ(ierr);
    t = t + deltat;

  }
  ierr = VecViewFromOptions(u,NULL,"-computed_sol");CHKERRQ(ierr);

  /*
     Compare with exact solution and show error norm
  */
  ierr = VecCopy(u,z);CHKERRQ(ierr);
  ierr = VecAXPY(z,-1.0,yex);CHKERRQ(ierr);
  ierr = VecNorm(z,NORM_2,&nrmd);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&nrmu);CHKERRQ(ierr);
  
  ierr = MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
  t2 = MPI_Wtime(); // end timer
  
  ierr = PetscPrintf(PETSC_COMM_WORLD," The relative error at t=%g is %.4f\n\n",(double)t,(double)(nrmd/nrmu));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",totits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of time steps: %D\n",steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Elapsed time is %f\n", t2 - t1 ); CHKERRQ(ierr);


  /*
     Free work space
  */
  ierr = MFNDestroy(&mfnexp);CHKERRQ(ierr);
  ierr = MFNDestroy(&mfnphi);CHKERRQ(ierr);
  ierr = MatDestroy(&L);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&yex);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}