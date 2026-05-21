import numpy as np
from model_solver import MacrophageModel, Params

class MacrophageModelNoHeaviside(MacrophageModel):
    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Overridden RHS function with customised dynamics.
        """

        M, N, p = self.p.M, self.p.N, self.p
        u = y.reshape((M + 1, N + 1))

        phi = self.compute_phi(u)
        VLt = self.compute_VL(u)


        du = np.zeros_like(u)

        if M >= 2:
            du[1:M, 0] = (
                - N * p.kplus * phi[1:M] * u[1:M, 0]
                + p.kminus * u[1:M, 1] 
                + M**2 * (phi[1:M] * (u[0:M-1, 0] + u[2:M+1, 0])
                    - u[1:M, 0] * (phi[0:M-1] + phi[2:M+1])
                )
                - p.beta * u[1:M, 0]
            )

            du[1:M, N] = (
                p.kplus * phi[1:M] * u[1:M, N-1]
                - N * p.kminus * u[1:M, N]
                + M**2 * p.D_min * (
                    phi[1:M] * (u[0:M-1, N] + u[2:M+1, N])
                    - u[1:M, N] * (phi[0:M-1] + phi[2:M+1])
                )
                - p.beta * u[1:M, N]
            )

        if N >= 2:
            n = np.arange(1, N)
            D_n = p.D_min + (1.0 - p.D_min) * self.n_decay[n]

            du[0, 1:N] = (
                N * p.kplus * phi[0] * (self.n_decay[n - 1] * u[0, 0:N-1] - self.n_decay[n] * u[0, 1:N])
                + N * p.kminus * (self.n_growth[n + 1] * u[0, 2:N+1] - self.n_growth[n] * u[0, 1:N])
                + M**2 * D_n * (phi[0] * u[1, 1:N] - phi[1] * u[0, 1:N])
                - p.beta * u[0, 1:N]
            )

            du[M, 1:N] = (
                N * p.kplus * phi[M] * (self.n_decay[n - 1] * u[M, 0:N-1] - self.n_decay[n] * u[M, 1:N])
                + N * p.kminus * (self.n_growth[n + 1] * u[M, 2:N+1] - self.n_growth[n] * u[M, 1:N])
                + M**2 * D_n * (phi[M] * u[M-1, 1:N] - phi[M-1] * u[M, 1:N] )
                - M * p.gamma * D_n * u[M, 1:N]
                - p.beta * u[M, 1:N]
            )

            if M >= 2:
                D_n_2d = p.D_min + (1.0 - p.D_min) * self.n_decay[n][None, :]
                du[1:M, 1:N] = (
                    N * p.kplus * phi[1:M, None] * (
                        self.n_decay[n - 1][None, :] * u[1:M, 0:N-1]
                        - self.n_decay[n][None, :] * u[1:M, 1:N]
                    )
                    + N * p.kminus * (
                        self.n_growth[n + 1][None, :] * u[1:M, 2:N+1]
                        - self.n_growth[n][None, :] * u[1:M, 1:N]
                    )
                    + M**2 * D_n_2d * (
                        phi[1:M, None] * (u[0:M-1, 1:N] + u[2:M+1, 1:N])
                        - u[1:M, 1:N] * (phi[0:M-1, None] + phi[2:M+1, None])
                    )
                    - p.beta * u[1:M, 1:N]
                )

        du[0, 0] = (
            - N * p.kplus * phi[0] * u[0, 0]
            + p.kminus * u[0, 1]
            + M * (p.sigma_b + p.sigma_max * VLt / (1.0 + VLt)) * phi[0]
            + M**2 * (phi[0] * u[1, 0] - phi[1] * u[0, 0])
            - p.beta * u[0, 0]
        )

        du[0, N] = (
            p.kplus * phi[0] * u[0, N-1]
            - N * p.kminus * u[0, N]
            + M**2 * p.D_min * (phi[0] * u[1, N] - phi[1] * u[0, N])
            - p.beta * u[0, N]
        )

        du[M, 0] = (
            - N * p.kplus * phi[M] * u[M, 0]
            + p.kminus * u[M, 1]
            + M**2 * (phi[M] * u[M-1, 0] - phi[M-1] * u[M, 0])
            - M * p.gamma * u[M, 0]
            - p.beta * u[M, 0]
        )

        du[M, N] = (
            p.kplus * phi[M] * u[M, N-1]
            - N * p.kminus * u[M, N]
            + M**2 * p.D_min * (phi[M] * u[M-1, N] - phi[M-1] * u[M, N])
            - M * p.gamma * p.D_min * u[M, N]
            - p.beta * u[M, N]
        )

        return du.ravel()
        
