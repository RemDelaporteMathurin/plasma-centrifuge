class PlasmaProperties:
    # Assumes n_tot is a constant for now
    def __init__(self, B, Ei, g, m_i, n_tot, alpha, **kwargs):

        self.B = B
        self.Ei = Ei * const.e # Convert eV to Joules
        self.g = g
        self.m_i = m_i
        self.n_tot = n_tot
        self.alpha = alpha

        # Store fundamental constants
        self.e = const.e
        self.k = const.k
        self.m_e = const.m_e
        self.h = const.h
        self.epsilon_0 = const.epsilon_0

        # Minimum ionization fraction constant
        self.min_frac = 0.0

        # Initialize interpolator attribute
        self.iz_interp = None

        iz_file_path = kwargs.get('iz_lookup_table')
        if iz_file_path:
            try:
                data = np.loadtxt(iz_file_path, comments='#', delimiter=',')
                densities, temps, vals = data[1:, 0], data[0, 1:], data[1:, 1:]
                self.iz_interp = RectBivariateSpline(densities, temps, vals)
            except Exception as e:
                print(f"Warning: Failed to load ionization table: {e}")

    def f(self, T_e):
        # Ionization fraction. If an ionization lookup table is provided, use
        # that. Otherwise, fall back to the Saha ionization equation.
        if self.iz_interp is not None:
            x = self.iz_interp(self.n_tot, T_e, grid=False)
        else:
            lambda_th_3 = (self.h / np.sqrt(2 * np.pi * self.m_e * self.k * T_e))**3
            A = 2 / (lambda_th_3 * self.n_tot) * self.g * np.exp(- self.Ei / (self.k * T_e))
            x = 0.5 * (np.sqrt(A**2 + 4 * A) - A)

        return self.min_frac + (1 - self.min_frac) * x

    def ni(self, T_e):
        # Number density of ions
        return self.f(T_e) * self.n_tot

    def ne(self, T_e):
        # Number density of electrons (quasineutrality)
        return self.ni(T_e)

    def omega_i(self):
        # Ion gyrofrequency
        return (self.e * self.B) / self.m_i

    def omega_e(self):
        # Electron gyrofrequency
        return (self.e * self.B) / self.m_e

    def lambda_D(self, T_e):
        # Debye length
        n_e = self.ne(T_e)
        debye_length = np.sqrt((self.epsilon_0 * self.k * T_e) / (n_e * self.e**2))
        return debye_length

    def coulomb_parameter(self, T_e):
        # Coulomb parameter
        n_e = self.ne(T_e)
        lambda_D = self.lambda_D(T_e)
        return 12 * np.pi * n_e * lambda_D**3

    def sigma_spitzer(self, T_e):
        # Spitzer conductivity
        ln_Lambda = np.log(self.coulomb_parameter(T_e))
        numerator = 4 * np.sqrt(2) / 3 * self.e**2 * np.sqrt(self.m_e) * ln_Lambda
        denominator = (4 * np.pi * self.epsilon_0)**2 * (self.k * T_e)**(1.5)
        return 1 / (numerator / denominator)

    def nu_ei(self, T_e):
        # Electron-ion collision frequency
        n_e = self.ne(T_e)
        conductivity = self.sigma_spitzer(T_e)
        return n_e * self.e**2 / (self.m_e * conductivity)

    def nu_en(self, T_e):
        """Electron-neutral collision frequency"""
        v_th_e = np.sqrt(3 * self.k * T_e / self.m_e)

        # Approximation of electron-neutral cross section
        Q_en = 5e-19  # m^2
        nn = (1 - self.f(T_e)) * self.n_tot
        return nn * v_th_e * Q_en

    def nu_in(self, T_h):
        # Ion-neutral collision frequency
        nn = (1 - self.f(T_h)) * self.n_tot
        v_th = np.sqrt(3 * self.k * T_h / self.m_i)
        return nn * v_th * self.Qin_Langevin(T_h)

    def beta_e(self, T_e):
        # Electron Hall parameter
        return self.omega_e() / self.nu_ei(T_e)

    def beta_i(self, T_h):
        # Ion Hall parameter
        return self.omega_i() / self.nu_in(T_h)

    def s(self, T_e, T_h):
        # Ion slip factor
        n_i = self.ne(T_e)
        return ((1 - self.f(T_e))**2) * self.beta_e(T_e) * self.beta_i(T_h)

    def beta_eff(self, T_e, T_h):
        # Effective Hall parameter
        s = self.s(T_e, T_h)
        beta_e = self.beta_e(T_e)
        numerator = s * (1 + s) + beta_e**2
        denominator = 1 + s
        return np.sqrt(numerator / denominator)

    def sigma_perp(self, T_e, T_h):
        # Perpendicular (Pedersen) conductivity
        return self.sigma_spitzer(T_e) / (1 + self.beta_eff(T_e, T_h)**2)

    def eta_perp(self, T_e, T_h):
        # Perpendicular (Pedersen) resistivity (defined for external operator usage)
        return 1 / self.sigma_perp(T_e, T_h)

    def sigma_H(self, T_e, T_h):
        # Hall conductivity
        beta_e = self.beta_e(T_e)
        return beta_e * self.sigma_spitzer(T_e) / ((1 + self.s(T_e, T_h))**2 + beta_e**2)

    def Qin_Langevin(self, T_h):
        # Ion-neutral cross section approximation using Langevin polarization scattering.
        # Depends on species polarizability, alpha. Useful when data is not readily available.
        v_th = np.sqrt(3 * self.k * T_h / self.m_i)  # Thermal velocity
        factor = np.sqrt(np.pi * self.alpha * self.e**2 / (self.epsilon_0 * self.m_i))
        return factor / v_th

    def mu_h(self, T_h):
        # Viscosity fit for hydrogen
        return -8.22278*10**(-6) + 5.03195*10**(-7) * T_h**0.58505

    def mu_h_prime(self, T_h):
        # Viscosity fit for hydrogen
        return 2.94394*10**(-7) / (T_h**0.41495)

    def kappa_h(self, T_h):
        # Thermal conductivity fit for hydrogen
        return -0.475048 + 0.0191871 * T_h**0.569845

    def kappa_e(self, T_e):
        """Braginskii equation for electron thermal conductivity perpendicular to B-field"""
        x = self.omega_e() / self.nu_ei(T_e)

        # Coefficients
        gamma0p, gamma1p = 11.92, 4.664
        delta0, delta1 = 3.7703, 14.79

        term1 = self.ne(T_e) * self.k**2 * T_e / (self.m_e * self.nu_ei(T_e))
        term2 = gamma1p * x**2 + gamma0p / (x**4 + delta1 * x**2 + delta0)
        return term1 * term2

    def ohmic_heating_h_perp(self, T_e, T_h):
        # Ohmic heating contribution to ions/heavy particles, MINUS the j^2 term
        s = np.full_like(T_e, 0.0) #self.s(T_e, T_h)
        return (s / (1 + s))**2 * self.sigma_perp(T_e, T_h) / self.mu_h(T_h)

    def ohmic_heating_h_parallel(self, T_e, T_h):
        # Ohmic heating contribution to ions/heavy particles, MINUS the j^2 term
        s = np.full_like(T_e, 0.0) #self.s(T_e, T_h)
        return (s / (1 + s))**2 * self.sigma_spitzer(T_e) / self.mu_h(T_h)

    def ohmic_heating_e_perp(self, T_e, T_h):
        # Ohmic heating contribution to electrons, MINUS the j^2 term
        s = np.full_like(T_e, 0.0) # self.s(T_e, T_h)
        return (1 / (1 + s))**2 / self.sigma_perp(T_e, T_h) / self.mu_h(T_h)

    def ohmic_heating_e_parallel(self, T_e, T_h):
        # Ohmic heating contribution to electrons, MINUS the j^2 term
        s = np.full_like(T_e, 0.0) # self.s(T_e, T_h)
        return (1 / (1 + s))**2 / self.sigma_spitzer(T_e) / self.mu_h(T_h)

    def e_h_energy_transfer(self, T_e, T_h):
        """Energy transfer rate per unit volume (W/m^3) between electrons and ions."""
        n_e = self.ne(T_e)
        nu_ei = self.nu_ei(T_e)
        nu_en = self.nu_en(T_e)
        return 3 * n_e * self.k * (T_e - T_h) * (self.m_e / self.m_i) * (nu_ei + nu_en)

    def heavy_particle_collisionality_coefficient(self, T_e, T_h):
        return 3 * self.ne(T_e) * const.k * const.m_e / (self.mu_h(T_h) * ion_mass) * (self.nu_ei(T_e) + self.nu_en(T_e)) * (T_e - T_h)

    def conductivity_ratio(self, T_e, T_h):
        return self.sigma_spitzer(T_e) / self.sigma_perp(T_e, T_h)
        
