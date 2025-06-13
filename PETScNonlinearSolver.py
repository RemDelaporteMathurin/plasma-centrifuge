class PETScNonlinearSolver:
    """Standard wrapper for PETSc.SNES."""

    def __init__(
        self,
        comm: MPI.Intracomm,
        problem: PETScNonlinearProblem,
        petsc_options: Optional[dict] = {},
        prefix: Optional[str] = None,
    ):
        self.b = fem.petsc.create_vector(problem.F_form)
        self.A = fem.petsc.create_matrix(problem.J_form)

        # Give PETSc solver options a unique prefix
        if prefix is None:
            prefix = f"snes_{str(id(self))[0:4]}"
        self.prefix = prefix
        self.petsc_options = petsc_options

        self.solver = PETSc.SNES().create(comm)
        self.solver.setOptionsPrefix(self.prefix)
        self.solver.setFunction(problem.F, self.b)
        self.solver.setJacobian(problem.J, self.A)
        self.set_petsc_options()
        self.solver.setFromOptions()

    def set_petsc_options(self):
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)

        for k, v in self.petsc_options.items():
            opts[k] = v

        opts.prefixPop()

    def solve(self, u: fem.Function) -> tuple[int, int]:
        self.solver.solve(None, u.x.petsc_vec)
        u.x.scatter_forward()
        return (self.solver.getIterationNumber(), self.solver.getConvergedReason())

    def __del__(self):
        self.solver.destroy()
        self.b.destroy()
        self.A.destroy()
