from scipy.io import loadmat
import mne

data = loadmat('ECoG_Handpose.mat')
print(data.keys())  # View available variable names


raw_mat = loadmat("ECoG_Handpose.mat")

y = data["y"]

print("Matrix shape:", y.shape)
print("First 5 values of row 0 (usually timestamps):", y[0, :5])
print("First 5 values of last row (usually triggers/labels):", y[-1, :5])

# Example: Assuming 'y' has shape (channels + 1, samples) where row 0 is time
eeg_data = raw_mat["y"][1:, :]  # Exclude timestamp row
sfreq = 256  # Replace with your g.tec device sampling rate (e.g., 256, 500, 1200 Hz)

# Convert microvolts to Volts (MNE expects Volts)
eeg_data_volts = eeg_data * 1e-6

# Create MNE Info and Raw objects
ch_names = [f"EEG {i+1}" for i in range(eeg_data.shape[0])]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(eeg_data_volts, info)

# Plot the signals
raw.plot()
