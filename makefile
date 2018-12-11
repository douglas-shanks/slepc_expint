CFLAGS     =
FFLAGS     =
CPPFLAGS   =
FPPFLAGS   =
LOCDIR     = src/mfn/examples/tutorials/
MANSEC     = MFN


include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

slepc_expint: slepc_expint.o chkopts
	-${CLINKER} -o slepc_expint slepc_expint.o ${SLEPC_MFN_LIB}
	${RM} slepc_expint.o
	
allclean:
	${RM} *.o slepc_expint
#------------------------------------------------------------------------------------

