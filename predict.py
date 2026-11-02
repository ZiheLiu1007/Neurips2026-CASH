import argparse
from pathlib import Path
import numpy as np
from runtime import add_prediction_arguments, prediction_inputs, predict


def main():
    parser = argparse.ArgumentParser(description="CASH")
    add_prediction_arguments(parser)
    args = parser.parse_args()
    model, x, r, indices, device = prediction_inputs(args)
    result = predict(model, x, r, indices["test"], args.fbar_lookback, device)
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "predictions.npz", **result)
    print(f'Saved {len(result["indices"])} forecasts to {folder}')


if __name__ == "__main__":
    main()
