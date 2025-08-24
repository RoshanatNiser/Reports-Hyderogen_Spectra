import numpy as np
import matplotlib.pyplot as plt
import re

# Constants for instrument conversion
msr = 0.0168  # degrees per minute of arc
vsr = 0.0084  # degrees per vernier scale division

def read_matrix(filename):
    """Read numerical data from file into matrix format"""
    try:
        with open(filename, 'r') as f:
            matrix = []
            for line in f:
                # Convert each line into a list of floats
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix
    except FileNotFoundError:
        print(f"File {filename} not found. Using sample data.")
        return None

def tot(M, V, O_vernier, dic=None, tit=None):
    """
    Convert measurements to total angles in degrees with corrected offset handling
    
    Parameters:
    -----------
    M : list - Main scale readings (degrees)
    V : list - Vernier scale readings (minutes)
    O_vernier : list - Offset for this specific vernier [MSR, VSR]
    dic : int - Direction indicator (kept for compatibility, not used)
    tit : str - Title for error reporting
    
    Returns:
    --------
    list - Corrected total angles in degrees
    """
    if len(M) != len(V):
        print(f"Error in {tit}: Lengths of M({len(M)}) and V({len(V)}) do not match.")
        return None
    
    T = []
    
    # Calculate offset for this vernier
    offset_degrees = msr * O_vernier[0] + vsr * O_vernier[1]
    
    for i in range(len(M)):
        # Convert reading to degrees
        reading_degrees = msr * M[i] + vsr * V[i]
        
        # Apply offset correction (subtract the zero reading)
        corrected_reading = reading_degrees - offset_degrees
        
        T.append(corrected_reading)
    
    return T

def theta(L_1, L_2, R_1, R_2):
    """Calculate diffraction angles from goniometer readings"""
    if len(L_1) != len(L_2) or len(R_1) != len(R_2) or len(L_1) != len(R_1):
        print('Array length mismatch in theta calculation')
        return None, None, None
    
    V_1 = []  # Left side angle differences
    V_2 = []  # Right side angle differences
    A = []    # Average diffraction angles
    
    for i in range(len(L_1)):
        V_1.append(L_1[i] - R_1[i])
        V_2.append(R_2[i] - L_2[i])
        # Average diffraction angle
        A.append((V_1[i] + V_2[i]) / 4)
    
    return V_1, V_2, A

def linear_least_squares(x_data, y_data):
    """Perform ordinary linear least squares fitting: y = a*x + b"""
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
    
    # Calculate residual variance
    y_pred = slope * x + intercept
    residuals = y - y_pred
    s_squared = np.sum(residuals**2) / (n - 2) if n > 2 else 0
    
    # Calculate parameter uncertainties
    var_denominator = S_xx - S_x**2/n
    if var_denominator > 0:
        slope_error = np.sqrt(s_squared / var_denominator)
        intercept_error = np.sqrt(s_squared * S_xx / (n * var_denominator))
        covariance = -s_squared * S_x / var_denominator
    else:
        slope_error = float('inf')
        intercept_error = float('inf')
        covariance = 0
    
    # Calculate R-squared
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - np.sum(residuals**2) / ss_tot if ss_tot > 0 else 0
    
    return {
        'slope': slope,
        'intercept': intercept,
        'slope_error': slope_error,
        'intercept_error': intercept_error,
        'covariance': covariance,
        'residual_variance': s_squared,
        'r_squared': r_squared,
        'residuals': residuals,
        'fitted_y': y_pred
    }

def weighted_least_squares(x_data, y_data, weights=None, y_errors=None):
    """Perform weighted linear least squares fitting"""
    x = np.array(x_data)
    y = np.array(y_data)
    n = len(x)
    
    if n != len(y):
        raise ValueError("x and y must have same length")
    if n < 3:
        raise ValueError("Need at least 3 data points for meaningful fit")
    
    # Determine weights
    if y_errors is not None:
        y_errors = np.array(y_errors)
        if len(y_errors) != n:
            raise ValueError("y_errors must have same length as data")
        # Handle zero or invalid errors
        valid_errors = y_errors > 0
        if not np.all(valid_errors):
            print("Warning: Some y_errors are zero or negative. Using equal weights.")
            weights = np.ones(n)
        else:
            weights = 1.0 / (y_errors**2)
    elif weights is not None:
        weights = np.array(weights)
        if len(weights) != n:
            raise ValueError("weights must have same length as data")
    else:
        weights = np.ones(n)
    
    # Calculate weighted sums
    S = np.sum(weights)
    S_x = np.sum(weights * x)
    S_y = np.sum(weights * y)
    S_xx = np.sum(weights * x**2)
    S_xy = np.sum(weights * x * y)
    
    # Calculate fitted parameters
    denominator = S * S_xx - S_x**2
    if abs(denominator) < 1e-12:
        raise ValueError("Denominator too small - data may be collinear")
        
    slope = (S * S_xy - S_x * S_y) / denominator
    intercept = (S_y - slope * S_x) / S
    
    # Calculate parameter uncertainties
    slope_error = np.sqrt(S / denominator)
    intercept_error = np.sqrt(S_xx / denominator)
    covariance = -S_x / denominator
    
    # Calculate weighted residuals and chi-squared
    y_pred = slope * x + intercept
    residuals = y - y_pred
    chi_squared = np.sum(weights * residuals**2)
    residual_variance = chi_squared / (n - 2) if n > 2 else 0
    
    # Calculate weighted R-squared
    y_mean_weighted = S_y / S
    ss_tot_weighted = np.sum(weights * (y - y_mean_weighted)**2)
    r_squared = 1 - chi_squared / ss_tot_weighted if ss_tot_weighted > 0 else 0
    
    return {
        'slope': slope,
        'intercept': intercept,
        'slope_error': slope_error,
        'intercept_error': intercept_error,
        'covariance': covariance,
        'residual_variance': residual_variance,
        'chi_squared': chi_squared,
        'r_squared': r_squared,
        'residuals': residuals,
        'fitted_y': y_pred,
        'weights': weights
    }

def predict_with_uncertainty(fit_result, x_new, x_error=0):
    """Predict y value at x_new with uncertainty propagation"""
    a = fit_result['slope']
    b = fit_result['intercept']
    sigma_a = fit_result['slope_error']
    sigma_b = fit_result['intercept_error']
    cov_ab = fit_result['covariance']
    
    # Predicted value
    y_pred = a * x_new + b
    
    # Check for valid uncertainties
    if np.isfinite(sigma_a) and np.isfinite(sigma_b) and np.isfinite(cov_ab):
        # Uncertainty from fit parameters
        var_from_fit = sigma_b**2 + x_new**2 * sigma_a**2 + 2 * x_new * cov_ab
        # Additional uncertainty from x_error
        var_from_x = (a * x_error)**2
        # Total uncertainty
        y_error = np.sqrt(max(0, var_from_fit + var_from_x))
    else:
        y_error = float('inf')
        var_from_fit = float('inf')
        var_from_x = 0
    
    return {
        'y_pred': y_pred,
        'y_error': y_error,
        'var_from_fit': var_from_fit,
        'var_from_x': var_from_x
    }




def plot_fit(x_data, y_data, fit_result, title="Linear Fit", 
             x_label="x", y_label="y", save_plot=True, dpi=300, 
             point_labels=None, error_bars=None):
    """Simple plotting function for data and fitted line"""
    x = np.array(x_data)
    y = np.array(y_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points with optional error bars
    if error_bars is not None:
        ax.errorbar(x, y, yerr=error_bars, fmt='bo', capsize=5, 
                   capthick=2, markersize=8, linewidth=2, label='Data')
    else:
        ax.scatter(x, y, color='blue', alpha=0.7, s=50, label='Data')
    
    # Plot fit line
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = fit_result['slope'] * x_fit + fit_result['intercept']
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fit')
    
    # Add point labels if provided
    if point_labels:
        for i, (xi, yi, label) in enumerate(zip(x, y, point_labels)):
            ax.annotate(label, (xi, yi), xytext=(10, 10), 
                       textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        filename = re.sub(r'[<>:"/\\|?*]', '_', title)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename.strip('_')
        plt.savefig(f"{filename}.png", dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as: {filename}.png")
    
    plt.show()
    return fig




def corrected_analysis():
    """Hydrogen Balmer series analysis with corrected offset handling"""
    print("=== Hydrogen Balmer Series Analysis (Corrected) ===\n")
    
    # Define offsets for each vernier (corrected approach)
    O_vernier1 = [0, 9]    # Vernier 1: 0°9'
    O_vernier2 = [180, 8]  # Vernier 2: 180°8'
    
    # Table 1 - Mercury calibration data
    print("Processing Table 1 (Mercury calibration)...")
    
    # Read actual files
    T_1_L_1_raw = read_matrix('T_1_L_1.txt')
    T_1_L_2_raw = read_matrix('T_1_L_2.txt')
    T_1_R_1_raw = read_matrix('T_1_R_1.txt')
    T_1_R_2_raw = read_matrix('T_1_R_2.txt')
    
    if any([x is None for x in [T_1_L_1_raw, T_1_L_2_raw, T_1_R_1_raw, T_1_R_2_raw]]):
        print("Error: Mercury data files not found!")
        return None, None, None, None
    
    # Process with corrected offset handling
    T_1_L_1 = tot(T_1_L_1_raw[0], T_1_L_1_raw[1], O_vernier1, tit='T_1_L_1')
    T_1_L_2 = tot(T_1_L_2_raw[0], T_1_L_2_raw[1], O_vernier2, tit='T_1_L_2')
    T_1_R_1 = tot(T_1_R_1_raw[0], T_1_R_1_raw[1], O_vernier1, tit='T_1_R_1')
    T_1_R_2 = tot(T_1_R_2_raw[0], T_1_R_2_raw[1], O_vernier2, tit='T_1_R_2')
    
    V_1, V_2, A_1 = theta(T_1_L_1, T_1_L_2, T_1_R_1, T_1_R_2)
    
    # Table 2 - Hydrogen data
    print("Processing Table 2 (Hydrogen data)...")
    
    T_2_L_1_raw = read_matrix('T_2_L_1.txt')
    T_2_L_2_raw = read_matrix('T_2_L_2.txt')
    T_2_R_1_raw = read_matrix('T_2_R_1.txt')
    T_2_R_2_raw = read_matrix('T_2_R_2.txt')
    
    if any([x is None for x in [T_2_L_1_raw, T_2_L_2_raw, T_2_R_1_raw, T_2_R_2_raw]]):
        print("Error: Hydrogen data files not found!")
        return None, None, None, None
    
    T_2_L_1 = tot(T_2_L_1_raw[0], T_2_L_1_raw[1], O_vernier1, tit='T_2_L_1')
    T_2_L_2 = tot(T_2_L_2_raw[0], T_2_L_2_raw[1], O_vernier2, tit='T_2_L_2')
    T_2_R_1 = tot(T_2_R_1_raw[0], T_2_R_1_raw[1], O_vernier1, tit='T_2_R_1')
    T_2_R_2 = tot(T_2_R_2_raw[0], T_2_R_2_raw[1], O_vernier2, tit='T_2_R_2')
    
    V_1_h, V_2_h, A_2 = theta(T_2_L_1, T_2_L_2, T_2_R_1, T_2_R_2)
    
    # Step 1: Mercury calibration
    print("\nStep 1: Mercury Calibration")
    print("-" * 30)
    
    # Read mercury wavelengths
    wavelength_data = read_matrix('Hg_wave.txt')
    if wavelength_data is None:
        print("Error: Mercury wavelength file not found!")
        return None, None, None, None
    
    wavelengths_hg = wavelength_data[0]  # Mercury wavelengths in nm
    
    # Ensure we have matching number of wavelengths and angles
    min_length = min(len(wavelengths_hg), len(A_1))
    wavelengths_hg = wavelengths_hg[:min_length]
    A_1 = A_1[:min_length]
    
    print(f"Mercury wavelengths (nm): {wavelengths_hg}")
    print(f"Mercury angles (degrees): {A_1}")
    
    # Convert angles to sin(theta) for mercury lines
    sin_theta_hg = np.sin(np.deg2rad(np.array(A_1)))
    print(f"sin(theta) values: {sin_theta_hg}")
    
    # Perform calibration fit
    try:
        hg_fit = linear_least_squares(sin_theta_hg, wavelengths_hg)
        
        print(f"\nCalibration Results:")
        print(f"Grating constant (slope): {hg_fit['slope']:.1f} nm")
        print(f"Intercept: {hg_fit['intercept']:.1f} nm")
        
        # Plot calibration
        plot_fit(sin_theta_hg, wavelengths_hg, hg_fit, 
                 title="Mercury Calibration", 
                 x_label="sin(θ)", 
                 y_label="Wavelength (nm)")
                 
    except Exception as e:
        print(f"Error in mercury calibration: {e}")
        return None, None, None, None
    
    # Step 2: Predict hydrogen wavelengths
    print(f"\nStep 2: Hydrogen Wavelength Prediction")
    print("-" * 40)
    
    # Hydrogen angles
    hydrogen_angles = np.array(A_2)
    sin_theta_h = np.sin(np.deg2rad(hydrogen_angles))
    
    print(f"Hydrogen angles (degrees): {hydrogen_angles}")
    print(f"Hydrogen sin(theta): {sin_theta_h}")
    
    # Angle measurement uncertainty
    theta_error_deg = 0.1  # degrees uncertainty
    theta_error_rad = np.deg2rad(theta_error_deg)
    
    # Predict wavelengths for each hydrogen line
    h_wavelengths = []
    h_uncertainties = []
    line_names = ['H-alpha (m=3)', 'H-beta (m=4)', 'H-gamma (m=5)']
    
    print(f"\nHydrogen line predictions:")
    for i, (angle, sin_th) in enumerate(zip(hydrogen_angles, sin_theta_h)):
        # Convert angle error to sin(theta) error
        cos_theta = np.cos(np.deg2rad(angle))
        sin_theta_error = cos_theta * theta_error_rad
        
        # Predict wavelength with uncertainty
        h_pred = predict_with_uncertainty(hg_fit, sin_th, sin_theta_error)
        
        h_wavelengths.append(h_pred['y_pred'])
        h_uncertainties.append(h_pred['y_error'])
        
        name = line_names[i] if i < len(line_names) else f'H-line {i+1}'
        print(f"{name}: {h_pred['y_pred']:.1f} ± {h_pred['y_error']:.1f} nm")
        print(f"  (Angle: {angle:.3f}°, sin(θ): {sin_th:.6f})")
    
    # Step 3: Rydberg constant fitting
    print(f"\nStep 3: Rydberg Constant Determination")
    print("-" * 38)
    
    # Prepare data for Rydberg fit
    m_values = [3, 4, 5][:len(h_wavelengths)]  # Principal quantum numbers for Balmer series
    x_rydberg = [(1/4 - 1/m**2) for m in m_values]
    
    # Convert wavelengths from nm to m and calculate 1/λ
    h_wavelengths_m = [lam * 1e-9 for lam in h_wavelengths]  # Convert nm to m
    y_rydberg = [1/lam for lam in h_wavelengths_m]  # 1/lambda (m^-1)
    
    # Propagate uncertainties: if y = 1/λ, then σ_y = σ_λ/λ²
    h_uncertainties_m = [err * 1e-9 for err in h_uncertainties]  # Convert nm to m
    y_errors = [err_m/(lam_m**2) for lam_m, err_m in zip(h_wavelengths_m, h_uncertainties_m)]
    
    print("Data for Rydberg fit:")
    for i, (m, x, y, yerr) in enumerate(zip(m_values, x_rydberg, y_rydberg, y_errors)):
        print(f"m={m}: x={x:.6f}, y={y:.2e} m⁻¹, σ_y={yerr:.2e} m⁻¹")
    
    # Perform weighted least squares fit
    try:
        ryd_fit = weighted_least_squares(x_rydberg, y_rydberg, y_errors=y_errors)
        
        print(f"\nRydberg Fit Results:")
        print(f"Rydberg constant: {ryd_fit['slope']:.3e} m⁻¹")
        print(f"Intercept: {ryd_fit['intercept']:.3e}")
        print(f"Chi-squared: {ryd_fit['chi_squared']:.3f}")
        
        # Compare with theoretical value
        theoretical_R = 1.097e7  # m^-1
        print(f"Theoretical Rydberg: {theoretical_R:.3e} m⁻¹")
        if np.isfinite(ryd_fit['slope']):
            relative_error = abs(ryd_fit['slope'] - theoretical_R)/theoretical_R * 100
            print(f"Relative error: {relative_error:.1f}%")
        
        # Plot Rydberg fit
        line_labels = ['Hα (m=3)', 'Hβ (m=4)', 'Hγ (m=5)'][:len(m_values)]
        plot_fit(x_rydberg, y_rydberg, ryd_fit, 
                 title="Rydberg Constant Determination",
                 x_label="(1/4 - 1/m²)", 
                 y_label="1/λ (m⁻¹)",
                 point_labels=line_labels,
                 error_bars=y_errors)
        
    except Exception as e:
        print(f"Error in Rydberg fit: {e}")
        ryd_fit = None
    
    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY FOR REPORT")
    print("="*50)
    print(f"Mercury Calibration:")
    print(f"  Grating constant: {hg_fit['slope']:.1f} nm")
    print(f"  Intercept: {hg_fit['intercept']:.2f} nm")
    print()
    print("Hydrogen wavelengths:")
    for i, name in enumerate(['H-alpha', 'H-beta', 'H-gamma'][:len(h_wavelengths)]):
        print(f"  {name}: {h_wavelengths[i]:.1f} ± {h_uncertainties[i]:.1f} nm")
    print()
    if ryd_fit:
        print(f"Rydberg constant: {ryd_fit['slope']:.3e} m⁻¹")
        print(f"Theoretical value: 1.097e+07 m⁻¹")
        if np.isfinite(ryd_fit['slope']):
            print(f"Relative error: {abs(ryd_fit['slope'] - 1.097e7)/1.097e7 * 100:.1f}%")
    
    return hg_fit, ryd_fit, h_wavelengths, h_uncertainties

if __name__ == "__main__":
    # Run the corrected analysis
    try:
        hg_result, ryd_result, h_wavelengths, h_uncertainties = corrected_analysis()
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Please check your data files and try again.")