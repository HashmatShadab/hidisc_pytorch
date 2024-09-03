import json
import matplotlib.pyplot as plt




if __name__ == '__main__':
    # Define the keys to plot

    # Use the function with labels and keys
    # keys_to_plot = ['train_clean_loss', 'train_adv_loss', 'train_clean_patient_loss', 'train_clean_slide_loss',
    #                 'train_clean_patch_loss', 'train_adv_patient_loss', 'train_adv_slide_loss', 'train_adv_patch_loss']

    import matplotlib.pyplot as plt
    import json
    import numpy as np

    keys_to_plot = ['train_adv_loss', 'train_adv_patient_loss', 'train_adv_slide_loss', 'train_adv_patch_loss']
    log_files = ["exp_19_logs.txt", "exp_24_logs.txt", "exp_25_logs.txt"]

    colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))



    for key in keys_to_plot:

        for i, log_file in enumerate(log_files):

            with open(log_file, 'r') as f:
                lines = f.readlines()

            values = []
            for line in lines:
                data = json.loads(line.strip())
                values.append(data[key])
            # label
            label = log_file.split(".")[0]
            plt.plot(values, label=label, color=colors[i])

        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f"{key}")
        plt.legend()
        plt.savefig(f"{key}.png")
        plt.show()



