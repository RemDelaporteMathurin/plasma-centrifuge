class NonlinearProblemWithCallback(NonlinearProblem):
    """Problem for the DOLFINx NewtonSolver with an external callback.

    It lets `NewtonSolver` to run an additional routine `external_callback`
    before vector and matrix assembly. This may be useful to perform additional
    calculations at each Newton iteration. In particular, external operators
    must be evaluated via this routine.
    """
    from typing import Callable, Optional

    def __init__(
        self,
        F: ufl.form.Form,
        u: fem.function.Function,
        bcs: list[fem.bcs.DirichletBC] = [],
        J: ufl.form.Form = None,
        form_compiler_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        external_callback: Optional[Callable] = lambda: None,
    ):
        super().__init__(F, u, bcs, J, form_compiler_options, jit_options)

        self.u = u
        self.external_callback = external_callback

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we also use it to evaluate the external operators.

        Args:
           x: The vector containing the latest solution
        """
        # The following line is from the standard NonlinearProblem class
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # We also copy the current solution values and propagate to the next step
        x.copy(self.u.x.petsc_vec)
        self.u.x.scatter_forward()

        # The external operators are evaluated here
        self.external_callback()
