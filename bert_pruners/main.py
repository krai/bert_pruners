import argparse
from .bert_pruner import BertPruner

def main():
    parser = argparse.ArgumentParser(description='BERT pruner')
    parser.add_argument('--saved_dir', type=str, help='Path to save the pruned model')
    parser.add_argument('--sparsity', type=float, help='Desired sparsity of the pruned model')
    args = parser.parse_args()

    pruner = BertPruner(saved_dir=args.saved_dir, sparsity=args.sparsity)
    pruner.prune_and_save()

if __name__ == "__main__":
    main()
