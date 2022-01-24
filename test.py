import torch
import numpy as np
import random
import argparse

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from glob import glob

from data import RobustDataset, AdaptiveResize, StatefulRandomCrop
from score import dice_score

def test(model_directory, data_glob, dataset_name):

    SEED = 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = 'cuda'
    batch_size = 16

    transform = Compose([AdaptiveResize((270, 480)), StatefulRandomCrop((256, 448))])
    dataset = RobustDataset(data_glob, common_transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=batch_size//2, pin_memory=True)

    model_names = sorted(glob(f"{model_directory}/*/model.pt"))
    models = [torch.jit.load(name, map_location=device) for name in model_names]
    model_count = len(models)

    scores_per_model = model_count * [None]
    ensemble_scores = None

    with torch.no_grad():
        for input, target in dataloader:

            input = input.to(device)
            target = target.to(device)

            target = torch.argmax(target, dim=1)

            output_sum = None

            for model_index, model in enumerate(models):

                output = model(input)[0]
    
                output_sum = output if output_sum is None else output_sum + output

                output = torch.argmax(output, dim=1)
                batch_scores = dice_score(output, target)
                scores_per_model[model_index] = batch_scores if scores_per_model[model_index] is None else torch.cat([scores_per_model[model_index], batch_scores])

            output = output_sum / model_count

            output = torch.argmax(output, dim=1)
            batch_scores = dice_score(output, target)
            ensemble_scores = batch_scores if ensemble_scores is None else torch.cat([ensemble_scores, batch_scores])

    model_scores = [torch.sum(scores).item() / torch.numel(scores) for scores in scores_per_model]
    ensemble_score = torch.sum(ensemble_scores).item() / torch.numel(ensemble_scores)

    # Saving/Printing Results.
    with open(f"{model_directory}/{dataset_name}_scores.txt", "w") as file:

        def log(output):
            file.write(output + "\n")
            print(output)

        for index, score in enumerate(model_scores):
            log(f"Cross Validation {index}: {score:.4f}")
        log("")
        log(f"Ensemble: {ensemble_score:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-directory", help="Directory to load the models from")
    parser.add_argument("-d", "--data-directory", help="Root directory for the robustmis dataset")
    parser.add_argument("-s", "--dataset", help="Dataset to test on (\"train\" or \"test\"")
    args = parser.parse_args()

    if not (args.dataset == "train" or args.dataset == "test"):
        raise argparse.ArgumentTypeError("Dataset must be \"test\" or \"train\"")

    data_glob = f"{args.data_directory}/{'Training/*/*/*' if args.dataset == 'train' else 'Testing/Stage_3/*/*/*'}"

    test(args.model_directory, data_glob, args.dataset)

