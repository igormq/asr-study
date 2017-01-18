import json
import codecs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    parser.add_argument('name', type=str,
                        help='dataset name')

    args = parser.parse_args()

    main(args.data_directory, os.path.join(DT_ABSPATH, 'cslu_spoltech_port'))
