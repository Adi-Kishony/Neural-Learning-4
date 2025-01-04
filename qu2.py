import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plotting_4_sample_neurons(target_t, target_activity, t, R, random_units):
    plt.figure(figsize=(10, 6))
    for i, unit in enumerate(random_units):
        plt.subplot(2, 2, i + 1)
        plt.plot(target_t, target_activity[unit], label='Target', color='blue')
        plt.plot(t, R[unit], label='Trained RNN', color='orange', alpha=0.7)
        plt.title(f'Unit {unit}')
        plt.xlabel('Time')
        plt.ylabel('Activity')
        plt.legend()
    plt.tight_layout()
    plt.show()

def units_before_and_after_training(t, R0, R, random_units):
    plt.figure(figsize=(10, 6))
    for i, unit in enumerate(random_units):
        plt.subplot(2, 2, i + 1)
        plt.plot(t, R0[unit], label='Untrained RNN', color='green', alpha=0.5)
        plt.plot(t, R[unit], label='Trained RNN', color='orange', alpha=0.7)
        plt.title(f'Unit {unit}')
        plt.xlabel('Time')
        plt.ylabel('Activity')
        plt.legend()
    plt.tight_layout()
    plt.show()

def heatmap_of_units_in_time(R_ds, R0_ds, target_activity):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    im1 = axs[0].imshow(target_activity, aspect='auto', cmap='viridis')
    axs[0].set_title('Target Activity')
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(R0_ds, aspect='auto', cmap='viridis')
    axs[1].set_title('Untrained RNN Activity')
    plt.colorbar(im2, ax=axs[1])

    im3 = axs[2].imshow(R_ds, aspect='auto', cmap='viridis')
    axs[2].set_title('Trained RNN Activity')
    plt.colorbar(im3, ax=axs[2])

    axs[2].set_xlabel('Time')
    for ax in axs:
        ax.set_ylabel('Units')
    plt.tight_layout()
    plt.show()

def plot_trajectories(data_proj, title, color, label, dims=(2, 3)):
    fig = plt.figure(figsize=(12, 6))

    # 2D plot
    if 2 in dims:
        plt.subplot(1, 2, 1)
        plt.plot(data_proj[0], data_proj[1], color=color, label=label)
        plt.scatter(data_proj[0, 0], data_proj[1, 0], color='red',
                    label='Start')  # Start point
        plt.scatter(data_proj[0, -1], data_proj[1, -1], color='black',
                    label='End')  # End point
        plt.title(f'{title} - 2D Trajectory')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()

    # 3D plot
    if 3 in dims:
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot(data_proj[0], data_proj[1], data_proj[2], color=color,
                label=label)
        ax.scatter(data_proj[0, 0], data_proj[1, 0], data_proj[2, 0],
                   color='red', label='Start')  # Start point
        ax.scatter(data_proj[0, -1], data_proj[1, -1], data_proj[2, -1],
                   color='black', label='End')  # End point
        ax.set_title(f'{title} - 3D Trajectory')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_explained_variance(target_var, R_var, R0_var):
    plt.figure(figsize=(10, 6))
    plt.plot(target_var, label="Target Activity", color="blue")
    plt.plot(R_var, label="Trained RNN", color="orange")
    plt.plot(R0_var, label="Untrained RNN", color="green")
    plt.axhline(0.9, color='red', linestyle='--', label='90% Variance')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of PCs")
    plt.ylabel("Explained Variance Ratio")
    plt.legend()
    plt.grid()
    plt.show()

def perform_pca(data, n_components=50):
    pca = PCA(n_components=n_components)
    pca_proj = pca.fit_transform(data.T).T  # Project data onto PC space
    explained_variance = pca.explained_variance_ratio_.cumsum()  # Cumulative explained variance
    return pca_proj, explained_variance, pca

def project_to_common_space(target_activity, R0_ds, trained_pca):
    target_common_proj = trained_pca.transform(target_activity.T).T
    R0_common_proj = trained_pca.transform(R0_ds.T).T
    return target_common_proj, R0_common_proj

if __name__ == '__main__':
    # Loading data
    D = hdf5storage.loadmat("ps4_data.mat")
    N = D['N']  # Number of RNN units
    target_t = D['target_t'].flatten()  # Time vector for target activity
    target_activity = D['target_activity']  # Target firing rates (N x len(target_t))
    t = D['t'].flatten()  # Time vector for RNN activity
    R = D['R']  # Trained RNN activity (N x len(t))
    R0 = D['R0']  # Untrained RNN activity (N x len(t))

    # 2
    np.random.seed(42)
    random_units = np.random.choice(N, 4, replace=False)
    # 2.1
    plotting_4_sample_neurons(target_t, target_activity, t, R, random_units)
    # 2.2
    units_before_and_after_training(t, R0, R, random_units)
    # 2.3
    step = len(t) // len(target_t)
    R_ds = R[:, ::step]
    R0_ds = R0[:, ::step]
    heatmap_of_units_in_time(R_ds, R0_ds, target_activity)

    # 2.2
    target_proj, target_var, target_pca = perform_pca(target_activity)
    R_proj, R_var, R_pca = perform_pca(R_ds)
    R0_proj, R0_var, R0_pca = perform_pca(R0_ds)
    # 2.2 1
    # Plot trajectories for each dataset
    plot_trajectories(target_proj, "Target Activity", color="blue",
                      label="Target")
    plot_trajectories(R_proj, "Trained RNN", color="orange", label="Trained")
    plot_trajectories(R0_proj, "Untrained RNN", color="green",
                      label="Untrained")

    # 2.2 2
    target_common_proj, R0_common_proj = project_to_common_space(
        target_activity, R_ds, R0_ds, R_pca)
    # Plot common space trajectories
    plot_trajectories(target_common_proj, "Target Activity (Common PCA)",
                      color="blue", label="Target")
    plot_trajectories(R_proj, "Trained RNN (Common PCA)", color="orange",
                      label="Trained")
    plot_trajectories(R0_common_proj, "Untrained RNN (Common PCA)",
                      color="green", label="Untrained")
    # 2.2 3
    plot_explained_variance(target_var, R_var, R0_var)





