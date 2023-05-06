import argparse

_PARSER = None


def get_default_parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = argparse.ArgumentParser()
    return _PARSER


def parse_args():
    parser = get_default_parser()
    args = parser.parse_args()
    return args


def add_trainer_args():
    parser = get_default_parser()
    parser.add_argument_group("trainer", "trainer arguments")
    parser.add_argument("--resume", action="store_true", help="resume training from the latest checkpoint")
    parser.add_argument("--log_steps", type=int, default=10, help="logging steps")
    parser.add_argument("--debug", action="store_true", help="debug mode with grad check")
    parser.add_argument("--detect_anomaly", action="store_true", help="detect anomaly with autograd")
    parser.add_argument("--save_latest_n_models", type=int, default=-1, help="save latest n models")
    parser.add_argument("--cudnn_deterministic", type=bool, default=True, help="use deterministic method")


def add_tester_args():
    parser = get_default_parser()
    parser.add_argument_group("tester", "tester arguments")
    parser.add_argument("--checkpoint", default=None, help="load from checkpoint")
    parser.add_argument("--test_epoch", type=int, default=None, help="test epoch")
    parser.add_argument("--cudnn_deterministic", type=bool, default=True, help="use deterministic method")
