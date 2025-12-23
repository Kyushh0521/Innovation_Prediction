from trainer import train
from utils import parse_args, load_config


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()