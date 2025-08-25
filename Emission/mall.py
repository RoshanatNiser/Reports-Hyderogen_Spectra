import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def linear_least_squares_full(x_data, y_data):
    """Enhanced linear least squares with full uncertainty analysis"""
    x = np.array(x_data)
    y = np.array(y_data)
    n = len(x)
    
    if n != len(y):
        raise ValueError("x and y must have same length")
    if n < 3:
        raise ValueError("Need at least 3 data points for meaningful fit")
    
    # Calculate sums
    S_x = np.sum(x)
    S_y = np.sum(y)
    S_xx = np.sum(x**2)
    S_xy = np.sum(x * y)
    
    # Calculate fitted parameters
    denominator = n * S_xx - S_x**2
    if abs(denominator) < 1e-12:
        raise ValueError("Denominator too small - data may be collinear")
        
    slope = (n * S_xy - S_x * S_y) / denominator
    intercept = (S_y - slope * S_x) / n
    
    # Calculate residuals and statistics
    y_pred = slope * x + intercept
    residuals = y - y_pred
    s_squared = np.sum(residuals**2) / (n - 2) if n > 2 else 0
    
    # Parameter uncertainties
    slope_error = np.sqrt(s_squared * n / denominator)
    intercept_error = np.sqrt(s_squared * S_xx / denominator)
    covariance = -s_squared * S_x / denominator
    
    # Correlation coefficient
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - np.sum(residuals**2) / ss_tot if ss_tot > 0 else 0
    
    return {
        'slope': slope,
        'intercept': intercept,
        'slope_error': slope_error,
        'intercept_error': intercept_error,
        'covariance': covariance,
        'residual_std': np.sqrt(s_squared),
        'r_squared': r_squared,
        'residuals': residuals,
        'fitted_y': y_pred,
        'n_points': n
    }

def propagate_wavelength_uncertainty(fit_result, lambda_obs, sigma_obs=0.2):
    """Propagate uncertainty for corrected wavelengths"""
    a = fit_result['slope']
    b = fit_result['intercept']
    sigma_a = fit_result['slope_error']
    sigma_b = fit_result['intercept_error']
    cov_ab = fit_result['covariance']
    
    # Corrected wavelength
    lambda_corr = a * lambda_obs + b
    
    # Uncertainty propagation
    var_from_fit = sigma_b**2 + lambda_obs**2 * sigma_a**2 + 2 * lambda_obs * cov_ab
    var_from_obs = (a * sigma_obs)**2
    sigma_corr = np.sqrt(var_from_fit + var_from_obs)
    
    return lambda_corr, sigma_corr

def wavelength_to_energy(wavelength_nm, sigma_wavelength_nm=0):
    """Convert wavelength to photon energy with uncertainty"""
    # Constants
    h = 4.135667696e-15  # eV·s
    c = 2.99792458e8     # m/s
    
    # Convert nm to m
    wavelength_m = wavelength_nm * 1e-9
    sigma_wavelength_m = sigma_wavelength_nm * 1e-9
    
    # Energy in eV
    energy = h * c / wavelength_m
    
    # Uncertainty in energy
    if sigma_wavelength_nm > 0:
        sigma_energy = energy * sigma_wavelength_m / wavelength_m
        return energy, sigma_energy
    else:
        return energy

def wavelength_to_wavenumber(wavelength_nm, sigma_wavelength_nm=0):
    """Convert wavelength to wavenumber with uncertainty"""
    # Convert nm to cm
    wavelength_cm = wavelength_nm * 1e-7
    sigma_wavelength_cm = sigma_wavelength_nm * 1e-7
    
    # Wavenumber in cm^-1
    wavenumber = 1 / wavelength_cm
    
    # Uncertainty in wavenumber
    if sigma_wavelength_nm > 0:
        sigma_wavenumber = wavenumber**2 * sigma_wavelength_cm
        return wavenumber, sigma_wavenumber
    else:
        return wavenumber

def analyze_spectroscopy_data():
    """Complete analysis of spectroscopy experiment data"""
    print("=== COMPLETE SPECTROSCOPY DATA ANALYSIS ===\n")
    
    # ============ MERCURY CALIBRATION DATA ============
    print("1. MERCURY CALIBRATION ANALYSIS")
    print("-" * 40)
    
    # Mercury data from image 4
    mercury_data = {
        'given_nm': [546.00, 579.0, 596.0, 615.0, 620.0, 623.0, 494.0, 435.0, 407.0, 403.00],
        'observed_nm': [546.05, 578.0, 599.0, 613.0, 620.0, 625.0, 495.0, 436.0, 408.0, 405.3],
        'colors': ['Green', 'Yellow', 'Orange', 'Red-3', 'Red_2', 'Red_1', 'Blue-bright', 'Indigo', 'Dim Violet', 'Bright Violet']
    }
    
    # Perform calibration fit
    cal_fit = linear_least_squares_full(mercury_data['observed_nm'], mercury_data['given_nm'])
    
    print(f"Calibration Results:")
    print(f"  Slope (a): {cal_fit['slope']:.6f} ± {cal_fit['slope_error']:.6f}")
    print(f"  Intercept (b): {cal_fit['intercept']:.3f} ± {cal_fit['intercept_error']:.3f} nm")
    print(f"  Correlation (R²): {cal_fit['r_squared']:.6f}")
    print(f"  Residual std: {cal_fit['residual_std']:.3f} nm")
    print(f"  Number of points: {cal_fit['n_points']}")
    
    # Create calibration plot
    plt.figure(figsize=(10, 6))
    
    plt.scatter(mercury_data['observed_nm'], mercury_data['given_nm'], 
                color='blue', s=60, alpha=0.7, label='Mercury lines')
    
    x_fit = np.linspace(min(mercury_data['observed_nm']), max(mercury_data['observed_nm']), 100)
    y_fit = cal_fit['slope'] * x_fit + cal_fit['intercept']
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Linear fit')
    
    # Add line labels
    for i, (obs, given, color) in enumerate(zip(mercury_data['observed_nm'], 
                                               mercury_data['given_nm'], 
                                               mercury_data['colors'])):
        plt.annotate(f'{given:.0f}', (obs, given), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    plt.xlabel('Observed Wavelength (nm)')
    plt.ylabel('Given Wavelength (nm)')
    plt.title(f'Mercury Calibration: λ_given = {cal_fit["slope"]:.4f}λ_obs + {cal_fit["intercept"]:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with error handling
    try:
        plt.savefig('mercury_calibration.png', dpi=300, bbox_inches='tight')
        print("  → Saved: mercury_calibration.png")
    except Exception as e:
        print(f"  Warning: Could not save mercury_calibration.png - {e}")
    
    plt.show()
    
    # ============ COPPER EMISSION ANALYSIS (ALL 12 LINES) ============
    print(f"\n2. COPPER EMISSION SPECTRUM ANALYSIS")
    print("-" * 40)
    
    # ALL 12 Copper data points
    copper_obs = [580.00, 571.10, 530.0, 521.0, 516.0, 510.0, 497.00, 464.0, 452.00, 451.0, 448.00, 427.0]
    copper_colors = ['Blue', 'Blue', 'Green', 'Green', 'Green', 'Green', 'Blue', 'Blue', 'Blue', 'Blue', 'Violet', 'Violet']
    
    # Apply calibration to copper data
    copper_results = []
    for i, obs in enumerate(copper_obs):
        corr, sigma = propagate_wavelength_uncertainty(cal_fit, obs)
        energy, energy_err = wavelength_to_energy(corr, sigma)
        copper_results.append({
            'observed': obs,
            'corrected': corr,
            'uncertainty': sigma,
            'energy_eV': energy,
            'energy_err': energy_err,
            'color': copper_colors[i]
        })
    
    print("Corrected Copper Lines (All 12):")
    for i, result in enumerate(copper_results):
        print(f"  Line {i+1:2d} ({result['color']:6s}): {result['corrected']:6.1f} ± {result['uncertainty']:4.1f} nm, "
              f"E = {result['energy_eV']:.3f} ± {result['energy_err']:.3f} eV")
    
    # ============ BRASS EMISSION ANALYSIS (ALL 14 LINES) ============
    print(f"\n3. BRASS EMISSION SPECTRUM ANALYSIS")
    print("-" * 40)
    
    # ALL 14 Brass data points
    brass_obs = [639.0, 580.0, 571.0, 529.0, 520.0, 515.0, 480.0, 472.0, 468.0, 464.0, 427.0, 438.0, 437.5, 436.8]
    
    # Apply calibration to brass data
    brass_results = []
    for obs in brass_obs:
        corr, sigma = propagate_wavelength_uncertainty(cal_fit, obs)
        energy, energy_err = wavelength_to_energy(corr, sigma)
        brass_results.append({
            'observed': obs,
            'corrected': corr,
            'uncertainty': sigma,
            'energy_eV': energy,
            'energy_err': energy_err
        })
    
    print("Corrected Brass Lines (All 14):")
    for i, result in enumerate(brass_results):
        element = "Zn" if result['corrected'] > 600 else "Cu"
        print(f"  Line {i+1:2d} ({element}): {result['corrected']:6.1f} ± {result['uncertainty']:4.1f} nm, "
              f"E = {result['energy_eV']:.3f} ± {result['energy_err']:.3f} eV")
    
    # Create metal emission spectra plot
    plt.figure(figsize=(12, 8))
    
    # Copper spectrum
    plt.subplot(2, 1, 1)
    copper_corr = [r['corrected'] for r in copper_results]
    copper_intensities = np.ones(len(copper_corr))  # Normalized intensities
    colors_cu = ['blue' if 'Blue' in r['color'] or 'Violet' in r['color'] 
                 else 'green' for r in copper_results]
    
    for i, (wl, intensity, color) in enumerate(zip(copper_corr, copper_intensities, colors_cu)):
        plt.vlines(wl, 0, intensity, colors=color, linewidth=3, alpha=0.7)
        if i < 10:  # Label first 10 lines to avoid clutter
            plt.text(wl, intensity + 0.05, f'{wl:.0f}', rotation=90, 
                    ha='center', va='bottom', fontsize=7)
    
    plt.xlim(400, 650)
    plt.ylim(0, 1.3)
    plt.ylabel('Relative Intensity')
    plt.title(f'Copper Emission Spectrum (12 lines)')
    plt.grid(True, alpha=0.3)
    
    # Brass spectrum
    plt.subplot(2, 1, 2)
    brass_corr = [r['corrected'] for r in brass_results]
    brass_intensities = np.ones(len(brass_corr))
    
    for i, (wl, intensity) in enumerate(zip(brass_corr, brass_intensities)):
        color = 'red' if wl > 600 else 'orange' if wl > 550 else 'green' if wl > 500 else 'blue'
        plt.vlines(wl, 0, intensity, colors=color, linewidth=3, alpha=0.7)
        if i < 12:  # Label first 12 lines
            plt.text(wl, intensity + 0.05, f'{wl:.0f}', rotation=90, 
                    ha='center', va='bottom', fontsize=7)
    
    plt.xlim(400, 650)
    plt.ylim(0, 1.3)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Relative Intensity')
    plt.title(f'Brass Emission Spectrum (14 lines)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with error handling
    try:
        plt.savefig('metal_emission_spectra.png', dpi=300, bbox_inches='tight')
        print("  → Saved: metal_emission_spectra.png")
    except Exception as e:
        print(f"  Warning: Could not save metal_emission_spectra.png - {e}")
    
    plt.show()
    
    # ============ IODINE ABSORPTION ANALYSIS (ALL 26 LINES) ============
    print(f"\n4. IODINE ABSORPTION SPECTRUM ANALYSIS")
    print("-" * 40)
    
    # ALL 26 Iodine data points
    iodine_obs = [608, 604, 599, 589, 586, 583, 582, 578, 570, 571, 580, 566, 563, 
                  544, 541, 539, 537, 534, 532, 562, 594, 575, 537, 545, 548, 565]
    
    # Apply calibration to iodine data
    iodine_results = []
    for obs in iodine_obs:
        corr, sigma = propagate_wavelength_uncertainty(cal_fit, obs)
        # Check for zero or negative wavelength
        if corr <= 0:
            print(f"Warning: Non-physical wavelength {corr} nm for observed {obs} nm")
            continue
        wavenumber, wave_sigma = wavelength_to_wavenumber(corr, sigma)
        iodine_results.append({
            'observed': obs,
            'corrected': corr,
            'uncertainty': sigma,
            'wavenumber': wavenumber,
            'wave_uncertainty': wave_sigma
        })
    
    # Sort by wavenumber (highest to lowest energy)
    iodine_results.sort(key=lambda x: x['wavenumber'], reverse=True)
    
    # Calculate wavenumber differences
    wavenumbers = [r['wavenumber'] for r in iodine_results]
    wave_diffs = []
    for i in range(1, len(wavenumbers)):
        diff = wavenumbers[i-1] - wavenumbers[i]
        wave_diffs.append(diff)
    
    # Calculate average spacing (ignore outliers)
    if wave_diffs:
        avg_spacing = np.mean(wave_diffs)
    else:
        avg_spacing = 0
    
    print("Iodine Vibronic Bands:")
    print(f"  Number of valid bands: {len(iodine_results)}")
    if iodine_results:
        print(f"  Wavelength range: {iodine_results[-1]['corrected']:.1f} - {iodine_results[0]['corrected']:.1f} nm")
        print(f"  Wavenumber range: {iodine_results[-1]['wavenumber']:.0f} - {iodine_results[0]['wavenumber']:.0f} cm⁻¹")
        if wave_diffs:
            print(f"  Average spacing: {avg_spacing:.0f} cm⁻¹")
        
        # Electronic transition energy (from lowest energy band)
        electronic_energy = wavelength_to_energy(iodine_results[0]['corrected'])
        print(f"  Electronic transition: {electronic_energy:.3f} eV")
        
        # Show first 15 bands
        print("\nFirst 15 Iodine Bands:")
        for i in range(min(15, len(iodine_results))):
            result = iodine_results[i]
            delta = wave_diffs[i] if i < len(wave_diffs) else 0
            print(f"  Band {i+1:2d}: {result['corrected']:6.1f} nm, {result['wavenumber']:5.0f} cm⁻¹, Δν̃ = {delta:4.0f} cm⁻¹")
    
    # Create iodine absorption spectrum plot
    if iodine_results:  # Only plot if we have valid data
        plt.figure(figsize=(12, 6))
        
        iodine_wavelengths = [r['corrected'] for r in iodine_results]
        iodine_wavenumbers = [r['wavenumber'] for r in iodine_results]
        
        # Show as absorption bands
        for i, wl in enumerate(iodine_wavelengths):
            plt.vlines(wl, 0, -1, colors='purple', linewidth=2, alpha=0.7)
            if i % 4 == 0:  # Label every 4th band to avoid clutter
                plt.text(wl, -0.8, f'{wl:.0f}', rotation=90, ha='center', va='top', fontsize=8)
        
        # Background continuum
        x_cont = np.linspace(min(iodine_wavelengths)-10, max(iodine_wavelengths)+10, 1000)
        y_cont = np.ones_like(x_cont)
        plt.plot(x_cont, y_cont, 'k-', linewidth=1, alpha=0.3, label='Continuum')
        
        plt.xlim(min(iodine_wavelengths)-20, max(iodine_wavelengths)+20)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorption')
        plt.title(f'Iodine Absorption Spectrum ({len(iodine_results)} bands)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot with error handling
        try:
            plt.savefig('iodine_absorption_spectrum.png', dpi=300, bbox_inches='tight')
            print("  → Saved: iodine_absorption_spectrum.png")
        except Exception as e:
            print(f"  Warning: Could not save iodine_absorption_spectrum.png - {e}")
        
        plt.show()
    
    # ============ SUMMARY FOR REPORT ============
    print(f"\n" + "="*60)
    print("SUMMARY FOR REPORT FILLING")
    print("="*60)
    
    print(f"\nCALIBRATION PARAMETERS:")
    print(f"  Slope (a) = {cal_fit['slope']:.6f} ± {cal_fit['slope_error']:.6f}")
    print(f"  Intercept (b) = {cal_fit['intercept']:.3f} ± {cal_fit['intercept_error']:.3f} nm")
    print(f"  R² = {cal_fit['r_squared']:.6f}")
    print(f"  Number of Hg lines = {len(mercury_data['given_nm'])}")
    print(f"  Residual standard deviation = {cal_fit['residual_std']:.3f} nm")
    
    print(f"\nCOPPER EMISSION LINES (12 total):")
    for i, result in enumerate(copper_results[:6]):  # Show first 6 for summary
        print(f"  Line {i+1} ({result['color']:6s}): {result['corrected']:6.1f} ± {result['uncertainty']:4.1f} nm, "
              f"Energy = {result['energy_eV']:.3f} ± {result['energy_err']:.3f} eV")
    print(f"  ... and {len(copper_results)-6} more lines (total {len(copper_results)})")
    
    print(f"\nBRASS EMISSION LINES (14 total):")
    for i, result in enumerate(brass_results[:6]):  # Show first 6 for summary
        element = "Zn" if result['corrected'] > 600 else "Cu"
        print(f"  Line {i+1} ({element}): {result['corrected']:6.1f} ± {result['uncertainty']:4.1f} nm, "
              f"Energy = {result['energy_eV']:.3f} ± {result['energy_err']:.3f} eV")
    print(f"  ... and {len(brass_results)-6} more lines (total {len(brass_results)})")
    
    if iodine_results:
        print(f"\nIODINE ABSORPTION ANALYSIS ({len(iodine_results)} bands):")
        print(f"  Number of vibronic bands = {len(iodine_results)}")
        print(f"  Electronic transition energy = {electronic_energy:.3f} eV")
        print(f"  Wavelength range = {iodine_results[-1]['corrected']:.1f} - {iodine_results[0]['corrected']:.1f} nm")
        print(f"  Wavenumber range = {iodine_results[-1]['wavenumber']:.0f} - {iodine_results[0]['wavenumber']:.0f} cm⁻¹")
        if wave_diffs:
            print(f"  Average vibrational spacing = {avg_spacing:.0f} cm⁻¹")
        print(f"  Typical wavelength uncertainty = ±{np.mean([r['uncertainty'] for r in iodine_results]):.1f} nm")
    
    # Generate comprehensive data tables for report
    create_comprehensive_tables(mercury_data, copper_results, brass_results, iodine_results, wave_diffs)
    
    return {
        'calibration': cal_fit,
        'copper': copper_results,
        'brass': brass_results,
        'iodine': iodine_results,
        'iodine_spacing': wave_diffs,
        'avg_spacing': avg_spacing,
        'electronic_energy': electronic_energy if iodine_results else None
    }

def create_comprehensive_tables(mercury_data, copper_results, brass_results, iodine_results, wave_diffs):
    """Create formatted tables with ALL data for the report"""
    
    print(f"\n" + "="*60)
    print("COMPREHENSIVE DATA TABLES FOR REPORT")
    print("="*60)
    
    print("\nTable 1: Mercury Calibration Data (10 points)")
    print("| S.No. | Given λ (nm) | Observed λ (nm) | Color |")
    print("|-------|--------------|------------------|--------|")
    
    for i, (given, obs, color) in enumerate(zip(mercury_data['given_nm'], 
                                               mercury_data['observed_nm'], 
                                               mercury_data['colors'])):
        print(f"| {i+1} | {given:.1f} | {obs:.1f} | {color} |")
    
    print("\nTable 2: Copper Emission Lines")
    print("| Line No. | Corrected λ (nm) | Uncertainty (nm) | Energy (eV) |")
    print("|----------|------------------|------------------|-------------|")
    
    for i, result in enumerate(copper_results):
        print(f"| {i+1} | {result['corrected']:.1f} | {result['uncertainty']:.1f} | {result['energy_eV']:.3f} |")
    
    print("\nTable 3: Brass Emission Lines")
    print("| Line No. | Element | Corrected λ (nm) | Energy (eV) |")
    print("|----------|---------|------------------|-------------|")
    
    for i, result in enumerate(brass_results):
        element = "Zn" if result['corrected'] > 600 else "Cu"
        print(f"| {i+1} | {element} | {result['corrected']:.1f} | {result['energy_eV']:.3f} |")
    
    if iodine_results:
        print(f"\nTable 4: Iodine Vibronic Absorption Bands ({len(iodine_results)} bands)")
        print("| Band No. | Corrected λ (nm) | Wavenumber (cm⁻¹) | Δν̃ (cm⁻¹) |")
        print("|----------|------------------|-------------------|------------|")
        
        for i in range(len(iodine_results)):
            result = iodine_results[i]
            delta = wave_diffs[i] if i < len(wave_diffs) else 0
            delta_str = f"{delta:.0f}" if delta != 0 else "-"
            print(f"| {i+1} | {result['corrected']:.1f} | {result['wavenumber']:.0f} | {delta_str} |")
    
    print(f"\nDATA SUMMARY:")
    print(f"- Mercury calibration: {len(mercury_data['given_nm'])} points")
    print(f"- Copper emission: {len(copper_results)} lines")
    print(f"- Brass emission: {len(brass_results)} lines") 
    print(f"- Iodine absorption: {len(iodine_results)} vibronic bands")

if __name__ == "__main__":
    # Run the complete analysis
    try:
        results = analyze_spectroscopy_data()
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- mercury_calibration.png")
        print("- metal_emission_spectra.png") 
        print("- iodine_absorption_spectrum.png")
        print(f"\nProcessed data counts:")
        print(f"- Copper: {len(results['copper'])} lines")
        print(f"- Brass: {len(results['brass'])} lines")
        print(f"- Iodine: {len(results['iodine'])} bands")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()