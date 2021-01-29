import argparse

from disentanglement_lib.config import reproduce


def main(
        study,
        model_num,
        model_name,
):
    # Obtain the study to reproduce.
    study = reproduce.STUDIES[study]

    config = study.get_model_config(model_num)[0]
    model = config[3]

    print(1 if model_name in model else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=str, default='unsupervised_study_v1')
    parser.add_argument('--model_name', type=str, default='beta')
    parser.add_argument('--model_num', type=int, default=0)
    args = parser.parse_args()

    main(
        study=args.study,
        model_num=args.model_num,
        model_name=args.model_name,
    )
