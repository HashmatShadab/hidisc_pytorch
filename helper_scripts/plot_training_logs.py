import matplotlib.pyplot as plt
import json
import numpy as np

if __name__ == '__main__':

    keys_to_plot = ['train_clean_loss', 'train_clean_patient_loss', 'train_clean_slide_loss', 'train_clean_patch_loss',
                    'train_adv_loss', 'train_adv_patient_loss', 'train_adv_slide_loss', 'train_adv_patch_loss']
    log_files = [r"F:\Code\Projects\hidisc_pytorch\Results\Baseline\resnet50_timm_pretrained_exp18\log.txt",
                 r"F:\Code\Projects\hidisc_pytorch\Results\Baseline\resnet50_exp18\log.txt",
                 r"F:\Code\Projects\hidisc_pytorch\Results\Baseline\resnet50_at_exp18\log.txt"]

    colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))


    def plot_loss_from_logs(keys_to_plot, log_files, colors):
        for key in keys_to_plot:
            plt.figure(figsize=(8, 6))  # Adjust figure size for better readability

            for i, log_file in enumerate(log_files):
                with open(log_file, 'r') as f:
                    values = [json.loads(line.strip())[key] for line in f.readlines()]

                label = log_file.split(".")[0]  # Extract label from filename
                plt.plot(values, color=colors[i], linewidth=2)  # Improved visibility

            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability

            # plt.savefig(f"{key}.png", dpi=300, bbox_inches='tight')  # Save with high resolution
            plt.show()  # Display plot


    plot_loss_from_logs(keys_to_plot, log_files, colors)


