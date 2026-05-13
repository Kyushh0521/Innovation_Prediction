from trainer import train
from utils import parse_args, load_config, set_seed


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    train(cfg, args)


if __name__ == "__main__":
    main()