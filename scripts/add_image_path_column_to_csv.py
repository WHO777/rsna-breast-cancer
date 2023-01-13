from pathlib import Path

from absl import app
from absl import flags

import pandas as pd

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string('csv_path', None, '')
    flags.DEFINE_string('images_dir', None, '')
    flags.DEFINE_string('output_csv_path', None, '')
    flags.DEFINE_string('image_path_column_name', 'image_path', '')


def main(_):
    csv_path = Path(FLAGS.csv_path).absolute()
    images_dir = Path(FLAGS.images_dir).absolute()
    output_csv_path = Path(FLAGS.output_csv_path).absolute()

    assert csv_path.is_file()
    assert images_dir.is_dir()

    df = pd.read_csv(csv_path)
    df[FLAGS.image_path_column_name] = df.apply(
        lambda i: str(images_dir / (str(i['patient_id']) + '_' + str(i['image_id']) + '.png')), axis=1
    )

    df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    define_flags()
    flags.mark_flag_as_required('csv_path')
    flags.mark_flag_as_required('images_dir')
    flags.mark_flag_as_required('output_csv_path')
    app.run(main)
