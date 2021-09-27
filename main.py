"""
Run all analyses
"""
from nlm import wrangle, mask, nsp, embeddings


def main():
    """Preprocess data & run all analyses"""
    wrangle.main()

    mask.main()
    nsp.main()
    embeddings.main()


if __name__ == "__main__":
    main()
