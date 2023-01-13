import random
from pathlib import Path

from absl import app
from absl import flags

from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string('csv_path', None, '')
    flags.DEFINE_integer('num_folds', 5, '')
    flags.DEFINE_string('column_split_name', 'cancer', '')
    flags.DEFINE_string('group_split_name', 'patient_id', '')
    flags.DEFINE_integer('val_fold_idx', None, '')
    flags.DEFINE_string('output_train_df_dir', None, '')
    flags.DEFINE_string('output_val_df_dir', None, '')
    flags.DEFINE_integer('seed', None, '')


def main(_):
    csv_path = FLAGS.csv_path

    assert Path(FLAGS.csv_path).is_file(), f'no such file {FLAGS.csv_path}.'

    output_train_df_dir = FLAGS.output_train_df_dir or '.'
    output_val_df_dir = FLAGS.output_val_df_dir or '.'

    val_fold_idx = FLAGS.val_fold_idx if FLAGS.val_fold_idx is not None \
        else random.randint(0, FLAGS.num_folds - 1)

    output_train_df_path = Path(output_train_df_dir) / f'train_fold_{val_fold_idx}.csv'
    output_val_df_path = Path(output_val_df_dir) / f'val_fold_{val_fold_idx}.csv'

    df = pd.read_csv(csv_path)

    sgkf = StratifiedGroupKFold(n_splits=FLAGS.num_folds, shuffle=True, random_state=FLAGS.seed)
    df['fold'] = -1

    for fold, (_, test_index) in enumerate(
        sgkf.split(df, df[FLAGS.column_split_name], df[FLAGS.group_split_name])
    ):
        df.loc[test_index, 'fold'] = fold

    train_df = df.query(f'fold != {val_fold_idx}')
    val_df = df.query(f'fold == {val_fold_idx}')

    train_df.to_csv(output_train_df_path, index=False)
    val_df.to_csv(output_val_df_path, index=False)


if __name__ == '__main__':
    define_flags()
    flags.mark_flag_as_required('csv_path')

    app.run(main)
