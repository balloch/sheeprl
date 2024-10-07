import argparse

from utils import compare_concepts


def main(embedding_path1, tp_path1, embedding_path2, tp_path2):
    compare_concepts(embedding_path1, tp_path1, embedding_path2, tp_path2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path1", type=str)
    parser.add_argument("--tp_path1", type=str)
    parser.add_argument("--embedding_path2", type=str)
    parser.add_argument("--tp_path2", type=str)
    args = parser.parse_args()
    main(args.embedding_path1, args.tp_path1, args.embedding_path2, args.tp_path2)
