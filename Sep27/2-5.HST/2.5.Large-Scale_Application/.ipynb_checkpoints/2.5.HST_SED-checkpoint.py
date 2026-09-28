import numpy as np
import os
import glob
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy import coordinates
import astropy.units as u
from astroquery.esa.hubble import ESAHubble
import matplotlib.pyplot as plt


def get_best_observation(c, filter_name, instrument='ACS/WFC'):
    """
    Finds the observation with the largest exposure duration for a given filter.
    Based on notebook sections 2.1 and source
    """
    log_step(f"Searching for observations with filter: {filter_name}...")
    
    # Cone search using user-defined parameters [cite: 7, 147]
    result = esahubble.cone_search_criteria(
        coordinates=c,
        radius=0.80*u.arcmin,
        obs_collection=['HST'],
        data_product_type='image',
        instrument_name=[instrument],
        filters=[filter_name],
        output_format="votable"
    )
    
    if len(result) == 0:
        log_step(f"No results found for filter {filter_name}.")
        return None
        
    log_step(f"Retrieved {len(result)} results. Identifying maximum exposure duration...")
    
    # Logic to select the longest exposure [cite: 8, 161]
    row_idx = np.argmax(result['exposure_duration'])
    max_row = result[row_idx]
    obs_id = max_row['observation_id']
    max_exp = max_row['exposure_duration']
    
    log_step(f"Selected Observation ID: {obs_id} (Exposure: {max_exp}s)")
    return obs_id

def process_observation(obs_id, ra, dec, filter_name):
    """
    Downloads the FITS file, extracts SCI/WHT data, and computes flux/error.
    Based on source [cite: 28, 29] and code logic for flux conversion.
    """
    log_step(f"Downloading FITS files for: {obs_id}...")
    esahubble.download_fits_files(observation_id=obs_id) # [cite: 28]
    
    # Locate the most recently downloaded FITS file [cite: 29, 653]
    fits_files = glob.glob('*.fits') + glob.glob('*.fits.gz')
    if not fits_files:
        log_step("Error: No FITS files found in directory.")
        return None
    latest_file = max(fits_files, key=os.path.getctime)
    
    log_step(f"Processing file: {latest_file}")
    
    with fits.open(latest_file) as hdul:
        # Extract data from SCI and WHT extensions 
        sci_data = hdul['SCI'].data
        wht_data = hdul['WHT'].data
        header = hdul['SCI'].header
        
        # Photometric keywords for flux conversion
        photflam = header['PHOTFLAM'] # Inverse sensitivity (erg/cm2/s/A per e-/s)
        photplam = header['PHOTPLAM'] # Pivot wavelength (Angstroms)
        
        # Convert RA/Dec to pixel coordinates using WCS
        wcs = WCS(header)
        px, py = wcs.world_to_pixel_values(ra, dec)
        ix, iy = int(np.round(px)), int(np.round(py))
        
        # Extract counts and weight at the specific pixel
        counts_per_sec = sci_data[iy, ix]
        weight = wht_data[iy, ix]
        
        # Calculate Flux and Error
        # Flux = Counts/sec * PHOTFLAM
        # Error = (1 / sqrt(WHT)) * PHOTFLAM
        flux = counts_per_sec * photflam
        error_flux = (1.0 / np.sqrt(weight)) * photflam
        
        log_step(f"Result for {filter_name}: Flux={flux:.2e}, Error={error_flux:.2e}, Pivot={photplam}")
        
        return {
            'Filter': filter_name,
            'Pivot_Wavelength_A': photplam,
            'Flux_erg_cm2_s_A': flux,
            'Error_Flux': error_flux,
            'Observation_ID': obs_id
        }

# Initialize the Hubble Archive interface
esahubble = ESAHubble()

def log_step(message, filename="hst_analysis_log.txt"):
    """Writes the current processing step to a text file for traceability."""
    with open(filename, "a") as f:
        f.write(message + "\n")
    print(message)

def run_hst_analysis(ra, dec, filter_list, instrument='ACS/WFC', output_csv="hst_sed_table.csv"):
    """
    Unified function to iterate through filters and compile the final dataset.
    Writes a CSV file and returns a DataFrame.
    """
    log_step(f"--- Starting Full Analysis for RA={ra}, Dec={dec} ---")
    c = coordinates.SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs') # [cite: 5]
    dataset = []
    
    for f in filter_list:
        log_step(f"\n--- Processing Filter: {f} ---")
        obs_id = get_best_observation(c, f, instrument)
        
        if obs_id:
            row_data = process_observation(obs_id, ra, dec, f)
            if row_data:
                dataset.append(row_data)
        else:
            log_step(f"Skipping filter {f} (no observations found).")
            
    # Create final table and save to computer
    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False)
    log_step(f"--- Analysis Complete. Data saved to {output_csv} ---")
    
    return df


# Example Usage:
target_ra, target_dec = 187.274535, 2.048654
filters_to_check = ['F606W', 'F814W', 'F435W', 'F475W', 'F555W', 'F625W', 'F775W']
sed_data = run_hst_analysis(target_ra, target_dec, filters_to_check)







##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################







def plot_hst_sed(dataset, log_filename="hst_analysis_log.txt"):
    """
    Plots the Spectral Energy Distribution (SED) from the retrieved HST data.
    
    Parameters:
    dataset (pd.DataFrame or list): The data containing 'Pivot_Wavelength_A', 
                                    'Flux_erg_cm2_s_A', and 'Error_Flux'.
    """
    with open(log_filename, "a") as log:
        log.write("Step: Initiating SED plot generation.\n")
    
    # Convert to DataFrame if it's a list for easier manipulation
    if isinstance(dataset, list):
        df = pd.DataFrame(dataset)
    else:
        df = dataset

    if df.empty:
        with open(log_filename, "a") as log:
            log.write("Warning: Dataset is empty. Plotting cancelled.\n")
        return

    # Sort by wavelength for a continuous line plot 
    df = df.sort_values('Pivot_Wavelength_A')

    plt.figure(figsize=(10, 6))
    
    # Plotting flux vs pivot wavelength with error bars 
    plt.errorbar(
        df['Pivot_Wavelength_A'], 
        df['Flux_erg_cm2_s_A'], 
        yerr=df['Error_Flux'], 
        fmt='o-', 
        capsize=5, 
        color='black', 
        markerfacecolor='blue', 
        label='HST Data'
    )

    # Annotate points with filter names for clarity
    for _, row in df.iterrows():
        plt.annotate(row['Filter'], (row['Pivot_Wavelength_A'], row['Flux_erg_cm2_s_A']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Log-log scale is standard for SED analysis
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel(r'Pivot Wavelength ($\AA$)')
    plt.ylabel(r'Flux ($erg \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$)')
    plt.title('HST Multi-Filter Spectral Energy Distribution')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    # Save the plot and log the success
    output_plot = "hst_sed_plot1.png"
    plt.savefig(output_plot)
    with open(log_filename, "a") as log:
        log.write(f"Step: SED plot saved to computer as {output_plot}.\n")
    
    plt.show()


plot_hst_sed(sed_data, log_filename="hst_analysis_log.txt")