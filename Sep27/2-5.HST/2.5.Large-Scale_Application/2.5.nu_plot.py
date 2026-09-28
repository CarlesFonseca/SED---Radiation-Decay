import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('hst_sed_table.csv')
wavelengths = df['Pivot_Wavelength_A']
flux = df['Flux_erg_cm2_s_A']
err_flux = df['Error_Flux']

c = 3e18 # Speed of light in Angstroms/second (approx)
# Using wavelengths in Angstroms to get frequency in Hz
nu = c / wavelengths
nu_fnu = wavelengths * flux
err_nu_fnu = wavelengths * err_flux # Propagate error linearly

# 3. Take the Logarithms
log_nu = np.log10(nu)
log_nu_fnu = np.log10(nu_fnu)

plt.figure(figsize=(8, 6))
plt.errorbar(log_nu, log_nu_fnu, 
             yerr=err_nu_fnu / (nu_fnu * np.log(10)), # Error propagation for log10
             fmt='o', capsize=5, linestyle='-', color='black', label='HST Data')

plt.xlabel(r'$\log_{10}(\nu \mathrm{[Hz]})$')
plt.ylabel(r'$\log_{10}(\nu F_\nu \mathrm{[erg\, s^{-1}\, cm^{-2}]})$')
plt.title('Spectral Energy Distribution (SED)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()