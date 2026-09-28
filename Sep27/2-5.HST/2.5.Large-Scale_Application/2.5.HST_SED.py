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
    Downloads the FITS file with retry logic and specific selection for 'drz' science files.
    """
    log_step(f"Downloading FITS files for: {obs_id}...")
    
    max_retries = 3
    success = False
    for attempt in range(max_retries):
        try:
            esahubble.download_fits_files(observation_id=obs_id)
            success = True
            break
        except (requests.exceptions.ChunkedEncodingError, Exception) as e:
            log_step(f"Download attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(5)
    
    if not success:
        log_step(f"Error: Failed to download {obs_id} after {max_retries} attempts.")
        return None

    # --- UPDATED FILE SELECTION LOGIC ---
    # 1. Get all fits files in the current directory
    all_fits = glob.glob('*.fits') + glob.glob('*.fits.gz')
    
    # 2. Filter for files that belong to THIS observation AND are science products (drz or drc)
    # We exclude 'trl', 'log', and 'asn' files which don't contain image data
    science_files = [
        f for f in all_fits 
        if obs_id in f and any(ext in f for ext in ['drz', 'drc'])
    ]
    
    if not science_files:
        # Fallback: if no drz/drc found, try to find any file with 'SCI' extension inside
        # by checking the most recent file that isn't a trailer/log
        log_step(f"Warning: No 'drz' or 'drc' file found for {obs_id}. Checking alternatives...")
        science_files = [f for f in all_fits if obs_id in f and '_trl' not in f and '_log' not in f]

    if not science_files:
        log_step(f"Error: No valid science FITS files found for {obs_id}.")
        return None

    # Pick the most recent valid science file
    latest_file = max(science_files, key=os.path.getctime)
    log_step(f"Processing science file: {latest_file}")
    # ------------------------------------

    try:
        with fits.open(latest_file) as hdul:
            # Verify extensions exist before accessing
            if 'SCI' not in hdul:
                log_step(f"Error: 'SCI' extension not found in {latest_file}. HDUs present: {hdul.info()}")
                return None
            
            sci_data = hdul['SCI'].data
            wht_data = hdul['WHT'].data
            header = hdul['SCI'].header
            
            photflam = header['PHOTFLAM']
            photplam = header['PHOTPLAM']
            
            wcs = WCS(header)
            px, py = wcs.world_to_pixel_values(ra, dec)
            ix, iy = int(np.round(px)), int(np.round(py))
            
            # Check bounds
            if iy >= sci_data.shape[0] or ix >= sci_data.shape[1] or iy < 0 or ix < 0:
                log_step(f"Error: Coordinates outside image bounds for {filter_name}.")
                return None

            counts_per_sec = sci_data[iy, ix]
            weight = wht_data[iy, ix]
            
            if weight <= 0:
                log_step(f"Warning: Zero weight at target location in {filter_name}.")
                return None

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
    except Exception as e:
        log_step(f"Error processing {latest_file}: {e}")
        return None

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
filters_to_check = ['F435W', 'F606W', 'F850LP', 'FR601N', 'FR782N'] # Obtained from discover_available_filters function
#Filter 'F606W' takes a very long time (skip in fast iterations)
# Filters that don't work: 'F814W', 'F475W', 'F555W', 'F625W', 'F775W', 'F105W', 'F547M'
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
