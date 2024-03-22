import json
import matplotlib.pyplot as plt


def plot_from_files(filenames, labels, keys):
    if len(filenames) != len(labels):
        print("Number of filenames does not match the number of labels.")
        return

    for key in keys:
        plt.figure(figsize=[10, 5])
        for filename, label in zip(filenames, labels):
            with open(filename, 'r') as f:
                data = [json.loads(line) for line in f]

            values = [item[key] for item in data]
            epochs = [item['Epoch'] for item in data]

            plt.plot(epochs, values, label=label)

        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend(loc='best')
        plt.title(f'{key} over Epochs')
        plt.savefig(f'{key}.png')
        plt.show()


# Use the function with labels and keys
keys_to_plot = ['train_clean_loss', 'train_adv_loss', 'train_clean_patient_loss', 'train_clean_slide_loss',
                'train_clean_patch_loss', 'train_adv_patient_loss', 'train_adv_slide_loss', 'train_adv_patch_loss']
# plot_from_files(['training_logs/Results/Baseline/resnet50_timm_pretrained_exp18/log.txt',
#                  "training_logs/Results/Baseline/resnet50_dynamic_aug_False/log.txt",
#                  "training_logs/Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/log.txt"],
#                 ['Natural Training (ImageNet pretrained)',
#                  'Natural Training (Scratch)',
#                  'Adv. Training (ImageNet Pretrained)'], keys_to_plot)

plot_from_files([
                 "training_logs/Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/log.txt",
"training_logs/Results/Adv/resnet50_attack_pgd_eps_8_dynaug_True_sanity_check_only_adv_loss/log.txt"],
                [
                 'ResNet-50 Adv. Training (ImageNet Pretrained) Warmup 5k',
                'ResNet-50 Adv. Training (Scratch) No Warmup'], keys_to_plot)

